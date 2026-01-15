from itertools import product
from tqdm import tqdm
from joblib import Parallel, delayed
import logging
from random import randrange

class ThreeSubsetMITM:
    
    def __init__(self, cipher, inputs, outputs, num_rounds, block_size, key_size, unit_size, num_pairs=64):
        
        logging.basicConfig(
            format='%(asctime)s [%(levelname)s] [%(funcName)s] %(message)s',
            level=logging.INFO
        )
        
        # Ensure the provided cipher has the required methods
        if not hasattr(cipher, "encrypt"):
            raise NameError(f"class {cipher} has no encrypt method")
        if not hasattr(cipher, "decrypt"):
            raise NameError(f"class {cipher} has no decrypt method")
        
        self.cipher = cipher
        self.num_rounds = num_rounds
        self.block_size = block_size
        self.key_size = key_size
        self.unit_size = unit_size
        
        if self.block_size % self.unit_size != 0:
            raise ValueError("unit size must divide block size")
        if self.key_size % self.unit_size != 0:
            raise ValueError("unit size must divide key size")
        
        self.num_block_units = self.block_size // self.unit_size
        self.num_key_units = self.key_size // self.unit_size
        self.unit_values = 2**self.unit_size
        
        if inputs is None or outputs is None:
            inputs, outputs, key = self.gen_inputs_outputs(num_rounds, num_pairs)
            logging.info(f"generated {num_pairs} input-output pairs with key {f"0x%0{self.key_size // 4}X"%(key)}")
        elif len(inputs) != len(outputs):
            raise ValueError("incompatible numbers of inputs and outputs")
        
        self.inputs = inputs
        self.outputs = outputs
    
    def get_unit(self, input, i):
        return (input >> self.unit_size*i) & (self.unit_values - 1)
    
    def gen_inputs_outputs(self, rounds, num_pairs):
        
        key = randrange(2**self.key_size)
        inputs = list()
        outputs = list()
        
        for _ in range(num_pairs):
            word = randrange(2**self.block_size)
            inputs.append(word)
            outputs.append(self.cipher.encrypt(word, key, rounds))
            
        return inputs, outputs, key
    
    def get_dependencies_helper(self, word, unit_size, rounds, decrypt):
    
        output_units = [
            [ [None for _ in range(self.unit_values)]
            for _ in range(self.num_block_units)
            ]
            for _ in range(self.num_key_units)
        ]

        output_depends_on = [set() for _ in range(self.num_block_units)]

        # i is the index of the key unit we change
        for i in range(self.num_key_units):
            # j is the value of this key unit
            for j in range(self.unit_values):
                
                if decrypt:
                    output = self.cipher.decrypt(word, j << unit_size*i, rounds, from_round=self.num_rounds)
                else:
                    output = self.cipher.encrypt(word, j << unit_size*i, rounds)
                    
                # k is the index of the output unit
                for k in range(self.num_block_units):
                    output_units[i][k][j] = self.get_unit(output, k)
                
        # Record which key units affect each output unit
        for k in range(self.num_block_units):
            for i in range(self.num_key_units):
                if len(set(output_units[i][k])) > 1:
                    output_depends_on[k].add(i)
            
        return output_depends_on
    
    def get_dependencies(self, rounds, decrypt=False, num_samples=1):
        """We determine which key units affect each output unit. This function uses the following fundamental procedure:
        
        Fix a word to encrypt. for a given key unit, we try all possible values, and record which output units fluctuate.
        
        We try this for several different random words, and for each output unit,
        we consider the intersection of all such sets as the overall set of affecting key units.
        
        Returns: a list of sets, each containing key units affecting the output unit of that index.
        """
        overall_deps = [set([_ for _ in range(self.num_key_units)]) for _ in range(self.num_block_units)]
        
        for _ in range(num_samples):
            word = randrange(2**self.block_size)
            deps = self.get_dependencies_helper(word, self.unit_size, rounds, decrypt)
            for k in range(self.num_block_units):
                overall_deps[k] &= deps[k]
                
        return overall_deps

    
    def find_attack(self, verbose=True):
        """Uses a greedy algorithm approach to find the best MITM attack for
        the provided number of rounds. Prints a summary of the attack if verbose=True.
        
        Returns: a dictionary giving attack parameters.
        
        Note: by best, we mean the attack with the lowest temporal complexity
        (we assume that the program is CPU-bound, not memory-bound).
        
        If we need to brute-force n key units, we expect to need at least n input-output pairs,
        or else we will start to get a lot of false positives. This function will display a
        warning if too few input-output pairs are detected.
        """
        enc_deps = list()
        dec_deps = list()
        eps = 1e-5
        
        for rounds in range(self.num_rounds + 1):
            enc_deps.append(self.get_dependencies(rounds))
            dec_deps.append(self.get_dependencies(rounds, decrypt=True))
        
        known_key_units = set()
        argmin_nonzero = lambda vals: min((i for i,v in enumerate(vals) if v != 0), key=lambda i: vals[i])
        schedule = list()
        
        while len(known_key_units) < self.num_key_units:
            attack_scores = list()
            
            # Assign each potential attack a score
            for enc_r in range(self.num_rounds + 1):
                for match_index in range(self.num_block_units):
                    K_c = (enc_deps[enc_r][match_index] & dec_deps[self.num_rounds - enc_r][match_index]) - known_key_units
                    K_enc = (enc_deps[enc_r][match_index] - K_c) - known_key_units
                    K_dec = (dec_deps[self.num_rounds - enc_r][match_index] - K_c) - known_key_units
                
                    # All else constant, favor attacks with more common key units
                    # (less memory complexity + easier parallelization)
                    attack_scores.append((1 - eps)*len(K_c) + max(len(K_enc), len(K_dec)))
            
            # Choose the attack with the smallest score
            try:
                opt_attack = argmin_nonzero(attack_scores)
            except:
                logging.error("not all key units influence the output")
                raise Exception()
            
            opt_enc_r = opt_attack // self.num_block_units
            opt_match_index = opt_attack % self.num_block_units
            
            K_c = (enc_deps[opt_enc_r][opt_match_index] & dec_deps[self.num_rounds - opt_enc_r][opt_match_index]) - known_key_units
            K_enc = (enc_deps[opt_enc_r][opt_match_index] - K_c) - known_key_units
            K_dec = (dec_deps[self.num_rounds - opt_enc_r][opt_match_index] - K_c) - known_key_units
            
            # Set parameters, including num_trials (the amount of input-output pairs we initially process per key)
            params = {"enc_r": opt_enc_r,
                      "dec_r": self.num_rounds - opt_enc_r,
                      "match_index": opt_match_index,
                      "num_trials": max(1, len(K_c) + max(len(K_enc), len(K_dec)) - 2),
                      "K_c": K_c,
                      "K_enc": K_enc,
                      "K_dec": K_dec}
            
            schedule.append(params)
            new_key_units = K_c | K_enc | K_dec
            if len(self.inputs) < len(new_key_units):
                logging.warning(f"small number of input-output pairs detected; at least {len(new_key_units)} is recommended")
            known_key_units.update(new_key_units)
        
        if verbose:
            logging.info("ATTACK SUMMARY:")
            for i, phase in enumerate(schedule):
                logging.info(f"Phase {i}: encrypt {phase["enc_r"]} rounds, decrypt {phase["dec_r"]} rounds," +
                    f" matching on unit {phase["match_index"]}, (|K_enc|, |K_dec|, |K_c|) = " +
                    f"{len(phase["K_enc"]), len(phase["K_dec"]), len(phase["K_c"])}")
            
        return schedule
    
    def execute_attack(self, schedule, parallel=True, progress=True):
        """Depth-first version of the attack"""
        
        poss_keys = [[None]*self.num_key_units]
        poss_keys = self.attack_helper(poss_keys, schedule, parallel, progress)
        
        logging.info(f"Attack finished! Total keys found: {len(poss_keys)}")
        return poss_keys
    
    def attack_helper(self, poss_keys, schedule, parallel, progress):
        
        if len(schedule) == 0:
            return poss_keys
        else:
            params = schedule[0]
        
        num_trials = params["num_trials"]
        K_enc = params["K_enc"]
        K_dec = params["K_dec"]
        K_c = params["K_c"]
        match_index = params["match_index"]
        enc_r = params["enc_r"]
        dec_r = params["dec_r"]
        
        new_poss_keys = list()

        # Consider each possible key separately
        for poss_key in poss_keys:
            enc_depends_on = K_enc.copy()
            dec_depends_on = K_dec.copy()
            com_depends_on = K_c.copy()
            
            # Record all known units and possibly discard them from the search space
            known_units = dict()
            for k in range(self.num_key_units):
                if poss_key[k] is not None:
                    known_units[k] = poss_key[k]
                    enc_depends_on.discard(k)
                    dec_depends_on.discard(k)
                    com_depends_on.discard(k)
                
            enc_depends_on = list(enc_depends_on)
            dec_depends_on = list(dec_depends_on)
            com_depends_on = list(com_depends_on)
            
            enc_num_deps = len(enc_depends_on)
            dec_num_deps = len(dec_depends_on)
            com_num_deps = len(com_depends_on)
            
            known_key_val = 0
            for j, val in known_units.items():
                known_key_val += val << self.unit_size*j
                
            def attack_with_common_units(common):
                job_poss_keys = list()
                
                # Prepare separate maps for both passes, for each input-output pair
                enc_map = dict()
                dec_map = dict()
                
                start_key_val = known_key_val
                for j in range(com_num_deps):
                    start_key_val += common[j] << self.unit_size*com_depends_on[j]
                
                # Iterate through each possible value for the forward pass key bytes
                for combo in product(range(self.unit_values), repeat=enc_num_deps):
                    
                    # Ensure the key is set with all the bytes we are guessing (and the ones we know)
                    key = start_key_val
                    packed_key = 0
                    for j in range(enc_num_deps):
                        key += combo[j] << self.unit_size*enc_depends_on[j]
                        packed_key |= (combo[j] << self.unit_size*j)
                        
                    # Save the resulting units to our map data structure
                    combined_ct = 0
                    for j, pt in enumerate(self.inputs[:num_trials]):
                        ct_enc = self.get_unit(self.cipher.encrypt(pt, key, enc_r), match_index)
                        combined_ct |= (ct_enc << self.unit_size*j)
                    
                    if combined_ct in enc_map:
                        enc_map[combined_ct].append(packed_key)
                    else:
                        enc_map[combined_ct] = [packed_key]

                # Iterate through each possible value for the backward pass key bytes
                for combo in product(range(self.unit_values), repeat=dec_num_deps):
                    
                    # Ensure the key is set with all the bytes we are guessing (and the ones we know)
                    key = start_key_val
                    packed_key = 0
                    for j in range(dec_num_deps):
                        key += combo[j] << self.unit_size*dec_depends_on[j]
                        packed_key |= (combo[j] << self.unit_size*j)
                        
                    # Save the resulting unit to our map data structure
                    combined_pt = 0
                    for j, ct in enumerate(self.outputs[:num_trials]):
                        pt_dec = self.get_unit(self.cipher.decrypt(ct, key, dec_r, self.num_rounds), match_index)
                        combined_pt |= (pt_dec << self.unit_size*j)
                    
                    if combined_pt in dec_map:
                        dec_map[combined_pt].append(packed_key)
                    else:
                        dec_map[combined_pt] = [packed_key]

                # Now search through all the keys we found to find a satisfactory collection of key units                
                # We iterate over the map with (presumably) fewer keys
                if dec_num_deps < enc_num_deps:
                    map_to_iterate = dec_map
                    map_to_search = enc_map
                    iterate_key_indices = dec_depends_on
                    search_key_indices = enc_depends_on
                    iterate_num_deps = dec_num_deps
                    search_num_deps = enc_num_deps
                else:
                    map_to_iterate = enc_map
                    map_to_search = dec_map
                    iterate_key_indices = enc_depends_on
                    search_key_indices = dec_depends_on
                    iterate_num_deps = enc_num_deps
                    search_num_deps = dec_num_deps
                            
                for (match_units, iterate_key_units_list) in map_to_iterate.items():
                    # Keep looking if we don't get an initial match
                    if match_units not in map_to_search:
                        continue
                    # Otherwise, for each of the keys producing a potential match, root out false positives
                    # by checking against all the input-output pairs we have
                    else:
                        search_key_units_list = map_to_search[match_units]
                        for iterate_key_units in iterate_key_units_list:
                            for search_key_units in search_key_units_list:
                                # Reconstruct the potential key
                                key = start_key_val
                                for j in range(iterate_num_deps):
                                    key += self.get_unit(iterate_key_units, j) << self.unit_size*iterate_key_indices[j]
                                for j in range(search_num_deps):
                                    key += self.get_unit(search_key_units, j) << self.unit_size*search_key_indices[j]
                                    
                                # Test the potential key against our entire remaining corpus
                                for (pt, ct) in list(zip(self.inputs, self.outputs))[num_trials:]:
                                    ct_enc = self.cipher.encrypt(pt, key, enc_r)
                                    pt_dec = self.cipher.decrypt(ct, key, dec_r, self.num_rounds)
                                    if self.get_unit(ct_enc, match_index) != self.get_unit(pt_dec, match_index):
                                        break
                                # If all the input-output pairs produce matches, record this as a possible key!
                                else:
                                    poss_key_copy = poss_key.copy()
                                    
                                    # Note the common units
                                    for j in range(com_num_deps):
                                        poss_key_copy[com_depends_on[j]] = common[j]
                                    
                                    # Note the units from the encryption and decryption keys
                                    for j in range(iterate_num_deps):
                                        poss_key_copy[iterate_key_indices[j]] = self.get_unit(iterate_key_units, j)
                                    for j in range(search_num_deps):
                                        poss_key_copy[search_key_indices[j]] = self.get_unit(search_key_units, j)
                                    
                                    # Print the resulting key (if we have a complete match)
                                    final_key = True
                                    for j in poss_key_copy:
                                        if j is None:
                                            final_key = False
                                    if final_key:
                                        integer_val = 0
                                        for i in range(self.num_key_units):
                                            integer_val += poss_key_copy[i] << 4*i
                                            
                                        # Check answer
                                        accurate_key = True
                                        for (pt, ct) in zip(self.inputs, self.outputs):
                                            ct_prime = self.cipher.encrypt(pt, integer_val, rounds=self.num_rounds)
                                            if ct_prime != ct:
                                                accurate_key = False
                                        
                                        if accurate_key:
                                            print(f"Key found, passes all tests: " + f"0x%0{self.key_size // 4}X"%(integer_val))
                                        else:
                                            print(f"Key found, fails a test: " + f"0x%0{self.key_size // 4}X"%(integer_val))
                                        
                                    job_poss_keys.extend(self.attack_helper([poss_key_copy], schedule[1:], parallel=False, progress=False))
                                    
                return job_poss_keys
                    
            # Iterate through each possible value for the overlapping key units
            combinations = product(range(self.unit_values), repeat=com_num_deps)
            
            if parallel:
                if progress:
                    all_results = Parallel(n_jobs=-1, verbose=0)(
                        delayed(attack_with_common_units)(common) for common in tqdm(combinations, total=self.unit_values**com_num_deps)
                    )
                else:
                    all_results = Parallel(n_jobs=-1, verbose=0)(
                        delayed(attack_with_common_units)(common) for common in combinations
                    )
                new_poss_keys.extend([item for sublist in all_results for item in sublist])
                
            else:
                if progress:
                    for common in tqdm(combinations, total=self.unit_values**com_num_deps):
                        new_poss_keys.extend(attack_with_common_units(common))
                else:
                    for common in combinations:
                        new_poss_keys.extend(attack_with_common_units(common))
                
        return new_poss_keys