from ThreeSubsetMITM import ThreeSubsetMITM

class PDBCC:
    
    def rotate_right(self, word, n, word_size=64):
        mask = 2**word_size - 1
        return ((word >> n) & mask) | ((word << (word_size - n) & mask)) 
    
    def rotate_left(self, word, n, word_size=64):
        mask = 2**word_size - 1
        return ((word << n) & mask) | (word >> (word_size - n) & mask)

    def next_keystate(self, keystate):
        return self.rotate_right(keystate ^ 0x3, 16, 64)
    
    def prev_keystate(self, keystate):
        return self.rotate_left(keystate, 16, 64) ^ 0x3

    def add_roundkey(self, word, keystate):
        return word ^ (keystate & 0xFFFFFFFF)

    def get_rows(self, word):
        row_0 = (word >> 48) & 0xFFFF 
        row_1 = (word >> 32) & 0xFFFF
        row_2 = (word >> 16) & 0xFFFF
        row_3 = (word >> 0) & 0xFFFF
        return row_0, row_1, row_2, row_3

    def mix_columns(self, c2):
        row_0, row_1, row_2, row_3 = self.get_rows(c2)
        c3 = row_0  << 48
        c3 |= (row_0 ^ row_1) << 32
        c3 |= (row_0 ^ row_1 ^ row_2) << 16
        c3 |= (row_0 ^ row_1 ^ row_2 ^ row_3)<< 0
        
        return c3
    
    def inv_mix_columns(self, c2):
        row_0, row_1, row_2, row_3 = self.get_rows(c2)
        c3 = row_0  << 48
        c3 |= (row_0 ^ row_1) << 32
        c3 |= (row_1 ^ row_2) << 16
        c3 |= (row_2 ^ row_3)<< 0
        
        return c3

    def rot_nib(self, c1):
        c2=(c1 << 16) | (c1 >> 48) & 0xFFFFFFFFFFFFFFFF
        return c2
    
    def inv_rot_nib(self, c1):
        c2=(c1 >> 16) | (c1 << 48) & 0xFFFFFFFFFFFFFFFF
        return c2

    def asbox(self, pt, sbox):
        c1 = 0
        for i in range(16):
            nib = (pt >> (i*4)) &  0xF
            c1 |= (sbox[nib] << i*4) 
        return c1

    def encrypt(self, plaintext, key, rounds=8):
        sbox = [0x0,0x3,0x5,0x8,0x6,0xA,0xF,0x4,0xE,0xD,0x9,0x2,0x1,0x7,0xC,0xB]
        
        for i in range(rounds):
            plaintext = self.asbox(plaintext, sbox)
            plaintext = self.rot_nib(plaintext)
            plaintext = self.mix_columns(plaintext)
            plaintext = self.add_roundkey(plaintext, key)
            key = self.next_keystate(key)    
        return plaintext
    
    def decrypt(self, ciphertext, key, rounds, from_round):
        inv_sbox = [0x0,0xc,0xb,0x1,0x7,0x2,0x4,0xd,0x3,0xa,0x5,0xf,0xe,0x9,0x8,0x6]
        
        for i in range(from_round):
            key = self.next_keystate(key)
        
        for i in range(rounds):
            key = self.prev_keystate(key)
            ciphertext = self.add_roundkey(ciphertext, key)
            ciphertext = self.inv_mix_columns(ciphertext)
            ciphertext = self.inv_rot_nib(ciphertext)
            ciphertext = self.asbox(ciphertext, inv_sbox)
            
        return ciphertext
    
if __name__ == "__main__":
        
    # 3F8CB6850C4D15E9
    pdbcc_data5 = [(0xDE6C41FD77A34823, 0xE81EDE37A9A69A1D),
                   (0xB6DDBF937AC4574E, 0xEC6E377AA4F0B357),
                   (0x00AD8E4DD84C7248, 0x395DB369A2E9C1F5),
                   (0xF474B7302BD77476, 0xD633D59EE05E3A55)
                   ]
    
    # 8917D1DA9031E779
    pdbcc_data6 = [(0x5E7C87E526EEF4B4, 0xC063C7349989591E),
                   (0xB1AF79104100B975, 0xEF01B559B67F68C4),
                   (0x0F74D2CDB8F8465A, 0x81E5FFC0AEE8376C),
                   (0x4ED0D5B0895BE743, 0x65E5ECAF94075967)
                   ]
    
    # E3204D3E8BD9B9E3
    pdbcc_data7 = [(0x5E3DD7EE8E1DA4EC, 0xDC4C0326A8688C66),
                   (0x5BD28D8CB614E4A1, 0x5D1819AED96206D6),
                   (0x63A7BE6A70A364BD, 0x914AB410048C8934),
                   (0x6DA923565700441E, 0x91D6B34D071C15FB),
                   (0x0DF29149730D18E2, 0x13D93C245AD83618),
                   (0x33973CFA6E1B3F5B, 0xB2B02136F4FAC743)]
    
    # 3AA95838E400CE5B
    pdbcc_data8 = [(0x51CEEECF6AF4B7DE, 0x4C3BB37F47EF5311),
                   (0xA27B230F1C94A9F7, 0xDAB289771BACA276),
                   (0x2FB8200FD1CD87A5, 0xD8EC61BB67786340),
                   (0xFFA2D4B639B3B1CF, 0x5FAD1E5ABEEBD271),
                   (0x9B17600A65E8D49A, 0xC0FA1C1F589370F5)]
    
    inputs, outputs = zip(*pdbcc_data5)
    
    pdbcc = PDBCC()
    mitm = ThreeSubsetMITM(pdbcc, inputs, outputs, num_rounds=5, block_size=64, key_size=64, unit_size=4)
    
    schedule = mitm.find_attack()
    keys = mitm.execute_attack(schedule, parallel=True)