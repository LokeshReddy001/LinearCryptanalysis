import random
import hashlib

sbox={0:0xE, 1:0x4, 2:0xD, 3:0x1, 4:0x2, 5:0xF, 6:0xB, 7:0x8, 8:0x3, 9:0xA, 0xA:0x6, 0xB:0xC, 0xC:0x5, 0xD:0x9, 0xE:0x0, 0xF:0x7}
sbox_inv = {0xE:0, 0x4:1, 0xD:2, 0x1:3, 0x2:4, 0xF:5, 0xB:6, 0x8:7, 0x3:8, 0xA:9, 0x6:0xA, 0xC:0xB, 0x5:0xC, 0x9:0xD, 0x0:0xE, 0x7:0xF}

pbox = {0:0, 1:4, 2:8, 3:12, 4:1, 5:5, 6:9, 7:13, 8:2, 9:6, 10:10, 11:14, 12:3, 13:7, 14:11, 15:15}

def initialise_sbox():
    global sbox, sbox_inv
    with open('sbox.txt', 'r') as f:
        sbox_str = f.read()

    # Convert comma-separated string to list of integers
    sbox_list = [int(x, 16) for x in sbox_str.split(',')]

    # Create dictionary from list
    sbox = {i: sbox_list[i] for i in range(len(sbox_list))}
    sbox_inv = {v: k for k, v in sbox.items()}
    print(sbox)

def keyGeneration():
    k = hashlib.sha1( hex(random.getrandbits(128)).encode('utf-8') ).hexdigest()[2:2+12]
    return k

def apply_sbox(state, sbox):
    sz = len(state)
    state = "0"*(4-sz)+state
    s1 = hex(sbox[int(state[0], 16)])[2:]
    s2 = hex(sbox[int(state[1], 16)])[2:]
    s3 = hex(sbox[int(state[2], 16)])[2:]
    s4 = hex(sbox[int(state[3], 16)])[2:]

    return s1+s2+s3+s4

def apply_pbox(state, pbox):
    x = int(state,16)
    state = bin(x)[2:]
    sz = len(state)
    state = "0"*(16-sz)+state
    per=""
    for i in range(16):
        per+=state[pbox[i]]
    return hex(int(per, 2))[2:]

def encrypt(pt, k):
    state = int(pt,16)
    
    subKeys = [ int(subK,16) for subK in [ k[0:4],k[4:8], k[8:12]] ]
    #ROUND 1
    #key mixing
    state = state^subKeys[0]
    #apply sbox
    state = hex(state)[2:]
    state = apply_sbox(state,sbox)
    #apply permutation
    state = apply_pbox(state,pbox)

    #ROUND 2
    state = int(state, 16)
    state = state^subKeys[1]
    #apply sbox
    state = hex(state)[2:]
    state = apply_sbox(state,sbox)
    #apply permutation
    # print(state)
    #ROUND 3
    state = int(state, 16)
    state = state^subKeys[2]
      
    return hex(state)[2:]


def decrypt(pt,k):

    state = int(pt,16)

    subKeys = [ int(subK,16) for subK in [ k[0:4],k[4:8], k[8:12]] ]

    state = state^subKeys[2]

    state = hex(state)[2:]
    state = apply_sbox(state,sbox_inv)

    state = int(state, 16)

    state = state^subKeys[1]
    state = hex(state)[2:]

    state = apply_pbox(state,pbox)
    state = apply_sbox(state,sbox_inv)

    state = int(state,16)
    state = state^subKeys[0]
    return hex(state)[2:]

import sys
import numpy as np
from math import trunc, fabs
import itertools as it
import collections

initialise_sbox()
sbox_in = ["".join(seq) for seq in it.product("01", repeat=4)]
# Build a table of output values
sbox_out = [ bin(sbox[int(seq,2)])[2:].zfill(4) for seq in sbox_in ]
# Build an ordered dictionary between input and output values
sbox_b = collections.OrderedDict(zip(sbox_in,sbox_out))
probBias = [[0 for x in range(len(sbox_b))] for y in range(len(sbox_b))] 
for bits in sbox_b.items():
    input_bits, output_bits = bits
    X1,X2,X3,X4 = [ int(bits,2) for bits in [input_bits[0],input_bits[1],input_bits[2],input_bits[3]] ]
    Y1,Y2,Y3,Y4 = [ int(bits,2) for bits in [output_bits[0],output_bits[1],output_bits[2],output_bits[3]] ]
                
    equations_in = [0, X4, X3, X3^X4, X2, X2^X4, X2^X3, X2^X3^X4, X1, X1^X4,
                    X1^X3, X1^X3^X4, X1^X2, X1^X2^X4, X1^X2^X3, X1^X2^X3^X4] 
                    
    equations_out = [0, Y4, Y3, Y3^Y4, Y2, Y2^Y4, Y2^Y3, Y2^Y3^Y4, Y1, Y1^Y4,
                    Y1^Y3, Y1^Y3^Y4, Y1^Y2, Y1^Y2^Y4, Y1^Y2^Y3, Y1^Y2^Y3^Y4]                
    
    for x_idx in range (0, len(equations_in)):
        for y_idx in range (0, len(equations_out)):
            probBias[x_idx][y_idx] += (equations_in[x_idx]==equations_out[y_idx])
probBias=[[(abs(probBias[i][j]-8) )for i in range(16)] for j in range(16)]

for j in probBias:
    print(j)

def find_indices(list_to_check, item_to_find):
    array = np.array(list_to_check)
    indices = np.where(array == item_to_find)[0]
    return list(indices)

if(len(sys.argv)==2):
    KEY=sys.argv[1]
elif len(sys.argv)==1:
    KEY = keyGeneration()
# print(KEY)

# K3_2= int(KEY[9],16)
# K3_3= int(KEY[10],16)
array = [[0 for i in range(16)] for j in range(4)]

def format_bin(x,n):
    state = bin(x)[2:]
    sz = len(state)
    state = "0"*(n-sz)+state
    return state

def format_hex(x):
    state = hex(x)[2:]
    sz = len(state)
    state = "0"*(4-sz)+state
    return state

N=10000

ciphertext=[0 for i in range(N)]

for i in range(N):
    hex_x = format_hex(i)

    ciphertext[i] = format_hex(int(encrypt(hex_x,KEY),16))
    

for i in range(N):

    hex_x = format_hex(i)
 
    temp=int(hex_x[1],16)&probBias[8].index(max(probBias[8]))
    p15=temp>>3
    p16=(temp>>2)&0b1
    p18=temp&0b1
    p17=(temp>>1)&0b1
    
    temp=int(hex_x[1],16)&probBias[4].index(max(probBias[4]))
    p25=temp>>3
    p26=(temp>>2)&0b1
    p28=temp&0b1
    p27=(temp>>1)&0b1

    temp=int(hex_x[1],16)&probBias[2].index(max(probBias[2]))
    p35=temp>>3
    p36=(temp>>2)&0b1
    p38=temp&0b1
    p37=(temp>>1)&0b1

    temp=int(hex_x[1],16)&probBias[1].index(max(probBias[1]))
    p45=temp>>3
    p46=(temp>>2)&0b1
    p48=temp&0b1
    p47=(temp>>1)&0b1
    # ciphertext[i] = format_hex(int(encrypt(hex_x,KEY),16))
    c1=int(ciphertext[i][0],16)
    c2=int(ciphertext[i][1],16)
    c3=int(ciphertext[i][2],16)
    c4=int(ciphertext[i][3],16)

    for k in range(16):
        u1=sbox_inv[c1^k]
        u2=sbox_inv[c2^k]
        u3=sbox_inv[c3^k]
        u4=sbox_inv[c4^k]

        u1_2=(u1>>2)&1
        u2_6=(u2>>2)&1
        u3_10=(u3>>2)&1
        u4_14=(u4>>2)&1

        res1=p15^p16^p17^p18^u1_2
        res2=(u2_6)^p25^p26^p27^p28
        res3=p35^p36^p37^p38^u3_10
        res4=p45^p46^p47^p48^u4_14

        if res1==0:
            array[0][k]+=1
        if res2==0:
            array[1][k]+=1
        if res3==0:
            array[2][k]+=1
        if res4==0:
            array[3][k]+=1


prob = [[0 for i in range(16)] for j in range(4)]

k3=[0 for i in range(4)]
round3_l=[]
for j in range(4):
    for i in range(16):
        prob[j][i]=abs(array[j][i]-(N/2))/N

    max_bias = max(prob[j])
    round3_l.append(find_indices(prob[j],max_bias))

# print(prob)
round3_keys=[]
# print(round3_l)
for k3_0 in round3_l[0]:
        # print(k3_0)
        for k3_1 in round3_l[1]:
            # print(k3_1)
            for k3_2 in round3_l[2]:
                # print(k3_2)
                for k3_3 in round3_l[3]:
                    # print(k3_3)
                    k1=hex(k3_0)[2:]
                    k2=hex(k3_1)[2:]
                    k3=hex(k3_2)[2:]
                    k4=hex(k3_3)[2:]
                    k3_calc=k1+k2+k3+k4
                    round3_keys.append(k3_calc)
                    # print("Calculated K3 ",k3_calc,", Actual K3 ",KEY[8:])

final_keys=[]

for k3_calc in round3_keys:

    #---------------------------------------------------------------------------------------------------------
    u2=[0 for i in range(N)]
    # N=10
    for i in range(N):
        temp = format_hex(int(ciphertext[i],16)^int(k3_calc, 16))
        temp0 = hex(sbox_inv[int(temp[0],16)])[2:]
        temp1 = hex(sbox_inv[int(temp[1],16)])[2:]
        temp2 = hex(sbox_inv[int(temp[2],16)])[2:]
        temp3 = hex(sbox_inv[int(temp[3],16)])[2:]
        u2[i]=  format_bin(int(temp0+temp1+temp2+temp3,16),16)

    #----------------------------------------------------------------------------------------------------------
    N1=5000
    bias_list1=[]
    bias_list2=[]
    for k in range(256):
        k2_guess=format_bin(k,8)
        count1=0
        count2=0
        for i in range(N1):

            p=format_bin(i,16)

            v1 = format_bin(int(k2_guess,2)^int(u2[i][1::2],2),8)
            v2 = format_bin(int(k2_guess,2)^int(u2[i][::2],2),8)
            # v1 = v1[::2]+v1[1::2]
            u1_5_8= format_bin(sbox_inv[int(v1[::2],2)],4)
            u1_13_16=format_bin(sbox_inv[int(v1[1::2],2)],4)
            u1_1_4= format_bin(sbox_inv[int(v2[::2],2)],4)
            u1_8_12=format_bin(sbox_inv[int(v2[1::2],2)],4)

            p_1_4=p[:4]
            p_5_8=p[4:8]
            p_8_12=p[8:12]
            p_13_16=p[12:]

            x = format_bin(int(u1_5_8,2)^int(p_5_8,2)^int(u1_13_16,2)^int(p_13_16,2),4)
            y = format_bin(int(u1_1_4,2)^int(p_1_4,2)^int(u1_8_12,2)^int(p_8_12,2),4)
            res1 = int(x[0],2)^int(x[1],2)^int(x[2],2)^int(x[3],2)
            res2 = int(y[0],2)^int(y[1],2)^int(y[2],2)^int(y[3],2)
            if(res1==0):
                count1+=1
            if(res2==0):
                count2+=1

        bias1 = abs((count1/N1) - 0.5)
        bias2 = abs((count2/N1) - 0.5)
        bias_list1.append(bias1)
        bias_list2.append(bias2)
    # print(bias_list1)
    # print("----------------------------------------")
    # print(bias_list2)
    # max_bias = max(bias_list)


    l1=find_indices(bias_list1,max(bias_list1))
    l2=find_indices(bias_list2,max(bias_list2))




    for c1 in l1:
        for c2 in l2:
  
            a1=int(format_bin(int(KEY[4:8],16),8)[1::2],2)
            a2=int(format_bin(int(KEY[4:8],16),8)[::2],2)
            # a=int(format_bin(int(KEY[4:8],16),16)[1::2],2)
            # print("Calculated round2 partial key:",c1)
            # print("Calculated round2 partial key:",c2)
            # print("Actual round2 key:",a1)
            # print("Actual round2 key:",a2)



            s1=format_bin(c1,8)
            s2=format_bin(c2,8)
            s=""
            for i in range(8):
                s+=s2[i]
                s+=s1[i]
            # print("Possible round2 key:",format_hex(int(s,2)))
            
            # print("Actual round2 key:",KEY[4:8])
            # print("--------------------------")
            k=int(u2[0],2)^int(s,2)
            k=apply_pbox(hex(k)[2:],pbox)
            k=apply_sbox(k,sbox_inv)
            # print("possible round1 key:",k)
            # print("actual round1 key:",KEY[:4])
            # print("--------------------------")
            final_keys.append(k+format_hex(int(s,2))+k3_calc)


    # print(bias_list1[a1],bias_list1[c1])
    # print(bias_list2[a2],bias_list2[c2])
    # print(bias_list[c],bias_list[a])
print("Actual Key: ",KEY)
print("---------------------------------------")
print("Possible final keys obtained")
for key in final_keys:
    print("|---",key)
print("                            ")
print("Verifying keys obtained....")
keys=[]

for key in final_keys:
    for i in range(N):
        if int(encrypt(hex(i)[2:],key),16)!=int(ciphertext[i],16):
            break
    else:
        print("|--",key)
        keys.append(key)
