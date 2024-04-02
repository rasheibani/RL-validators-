[define indoor_room]
indoor_room is a kind of room.
indoor_room has a text called description.
indoor_room has a text called printed name.

[define area]
area is a kind of room.
area has a text called description.
area has a text called printed name.
area has a indoor_room called parent.
area has a text called X.
area has a text called Y.
area can be enterable.
area is always enterable.

player has a number called orientation.


[define landmark]
landmark is a kind of thing.
landmark has a text called description.
landmark has a text called printed name.
landmark can be examined.
area has a list of landmark called visible_objects.

[create r0]
r0 is a indoor_room. "Room 0".

[create r1]
r1 is a indoor_room. "Room 1".

[create r2]
r2 is a indoor_room. "Room 2".

[create r3]
r3 is a indoor_room. "Room 3".

[create r4]
r4 is a indoor_room. "Room 4".

[create r5]
r5 is a indoor_room. "Room 5".

[create r6]
r6 is a indoor_room. "Room 6".

[create a24r0]
a24r0 is a area. "An area (24) in r0".
the printed name of the a24r0 is "Room 0".
Understand "Area 24 in Room 0" as a24r0.
the parent of the a24r0 is r0.
the X of the a24r0 is "547288.0".
the Y of the a24r0 is "1034172.0".

[create a36r0]
a36r0 is a area. "An area (36) in r0".
the printed name of the a36r0 is "Room 0".
Understand "Area 36 in Room 0" as a36r0.
the parent of the a36r0 is r0.
the X of the a36r0 is "547570.0".
the Y of the a36r0 is "1034200.0".

[create a37r0]
a37r0 is a area. "An area (37) in r0".
the printed name of the a37r0 is "Room 0".
Understand "Area 37 in Room 0" as a37r0.
the parent of the a37r0 is r0.
the X of the a37r0 is "547529.2968536067".
the Y of the a37r0 is "1034251.4209743899".

[create a44r0]
a44r0 is a area. "An area (44) in r0".
the printed name of the a44r0 is "Room 0".
Understand "Area 44 in Room 0" as a44r0.
the parent of the a44r0 is r0.
the X of the a44r0 is "547318.8816727516".
the Y of the a44r0 is "1034219.4952958586".

[create a59r0]
a59r0 is a area. "An area (59) in r0".
the printed name of the a59r0 is "Room 0".
Understand "Area 59 in Room 0" as a59r0.
the parent of the a59r0 is r0.
the X of the a59r0 is "547422.0086022716".
the Y of the a59r0 is "1034364.0104014394".

[create a60r0]
a60r0 is a area. "An area (60) in r0".
the printed name of the a60r0 is "Room 0".
Understand "Area 60 in Room 0" as a60r0.
the parent of the a60r0 is r0.
the X of the a60r0 is "547435.7496886395".
the Y of the a60r0 is "1034362.317373606".

[create a20r1]
a20r1 is a area. "An area (20) in r1".
the printed name of the a20r1 is "Room 1".
Understand "Area 20 in Room 1" as a20r1.
the parent of the a20r1 is r1.
the X of the a20r1 is "548176.0".
the Y of the a20r1 is "1036304.0".

[create a21r1]
a21r1 is a area. "An area (21) in r1".
the printed name of the a21r1 is "Room 1".
Understand "Area 21 in Room 1" as a21r1.
the parent of the a21r1 is r1.
the X of the a21r1 is "548142.6666666666".
the Y of the a21r1 is "1036253.5364495007".

[create a40r1]
a40r1 is a area. "An area (40) in r1".
the printed name of the a40r1 is "Room 1".
Understand "Area 40 in Room 1" as a40r1.
the parent of the a40r1 is r1.
the X of the a40r1 is "547976.0".
the Y of the a40r1 is "1036304.0".

[create a41r1]
a41r1 is a area. "An area (41) in r1".
the printed name of the a41r1 is "Room 1".
Understand "Area 41 in Room 1" as a41r1.
the parent of the a41r1 is r1.
the X of the a41r1 is "548009.333333333".
the Y of the a41r1 is "1036260.9694444445".

[create a51r1]
a51r1 is a area. "An area (51) in r1".
the printed name of the a51r1 is "Room 1".
Understand "Area 51 in Room 1" as a51r1.
the parent of the a51r1 is r1.
the X of the a51r1 is "548076.5167685037".
the Y of the a51r1 is "1036173.513395759".

[create a63r1]
a63r1 is a area. "An area (63) in r1".
the printed name of the a63r1 is "Room 1".
Understand "Area 63 in Room 1" as a63r1.
the parent of the a63r1 is r1.
the X of the a63r1 is "548081.8708448358".
the Y of the a63r1 is "1036005.3162913685".

[create a38r2]
a38r2 is a area. "An area (38) in r2".
the printed name of the a38r2 is "Room 2".
Understand "Area 38 in Room 2" as a38r2.
the parent of the a38r2 is r2.
the X of the a38r2 is "548833.0".
the Y of the a38r2 is "1034200.0".

[create a39r2]
a39r2 is a area. "An area (39) in r2".
the printed name of the a39r2 is "Room 2".
Understand "Area 39 in Room 2" as a39r2.
the parent of the a39r2 is r2.
the X of the a39r2 is "548792.4539705467".
the Y of the a39r2 is "1034248.7643334025".

[create a45r2]
a45r2 is a area. "An area (45) in r2".
the printed name of the a45r2 is "Room 2".
Understand "Area 45 in Room 2" as a45r2.
the parent of the a45r2 is r2.
the X of the a45r2 is "548550.0".
the Y of the a45r2 is "1034172.0".

[create a46r2]
a46r2 is a area. "An area (46) in r2".
the printed name of the a46r2 is "Room 2".
Understand "Area 46 in Room 2" as a46r2.
the parent of the a46r2 is r2.
the X of the a46r2 is "548580.7043713275".
the Y of the a46r2 is "1034222.7067112251".

[create a57r2]
a57r2 is a area. "An area (57) in r2".
the printed name of the a57r2 is "Room 2".
Understand "Area 57 in Room 2" as a57r2.
the parent of the a57r2 is r2.
the X of the a57r2 is "548676.2390740268".
the Y of the a57r2 is "1034362.4312827691".

[create a58r2]
a58r2 is a area. "An area (58) in r2".
the printed name of the a58r2 is "Room 2".
Understand "Area 58 in Room 2" as a58r2.
the parent of the a58r2 is r2.
the X of the a58r2 is "548698.2896516462".
the Y of the a58r2 is "1034360.8833195835".

[create a2r3]
a2r3 is a area. "An area (2) in r3".
the printed name of the a2r3 is "Room 3".
Understand "Area 2 in Room 3" as a2r3.
the parent of the a2r3 is r3.
the X of the a2r3 is "548827.0".
the Y of the a2r3 is "1036314.0".

[create a3r3]
a3r3 is a area. "An area (3) in r3".
the printed name of the a3r3 is "Room 3".
Understand "Area 3 in Room 3" as a3r3.
the parent of the a3r3 is r3.
the X of the a3r3 is "548857.9153780069".
the Y of the a3r3 is "1036288.1834621994".

[create a4r3]
a4r3 is a area. "An area (4) in r3".
the printed name of the a4r3 is "Room 3".
Understand "Area 4 in Room 3" as a4r3.
the parent of the a4r3 is r3.
the X of the a4r3 is "549586.0".
the Y of the a4r3 is "1036314.0".

[create a5r3]
a5r3 is a area. "An area (5) in r3".
the printed name of the a5r3 is "Room 3".
Understand "Area 5 in Room 3" as a5r3.
the parent of the a5r3 is r3.
the X of the a5r3 is "549554.9123931625".
the Y of the a5r3 is "1036288.1662393163".

[create a12r3]
a12r3 is a area. "An area (12) in r3".
the printed name of the a12r3 is "Room 3".
Understand "Area 12 in Room 3" as a12r3.
the parent of the a12r3 is r3.
the X of the a12r3 is "548818.0".
the Y of the a12r3 is "1036224.0".

[create a13r3]
a13r3 is a area. "An area (13) in r3".
the printed name of the a13r3 is "Room 3".
Understand "Area 13 in Room 3" as a13r3.
the parent of the a13r3 is r3.
the X of the a13r3 is "548858.7622699386".
the Y of the a13r3 is "1036242.6487730059".

[create a14r3]
a14r3 is a area. "An area (14) in r3".
the printed name of the a14r3 is "Room 3".
Understand "Area 14 in Room 3" as a14r3.
the parent of the a14r3 is r3.
the X of the a14r3 is "549595.0".
the Y of the a14r3 is "1036224.0".

[create a15r3]
a15r3 is a area. "An area (15) in r3".
the printed name of the a15r3 is "Room 3".
Understand "Area 15 in Room 3" as a15r3.
the parent of the a15r3 is r3.
the X of the a15r3 is "549554.662577747".
the Y of the a15r3 is "1036242.6912577746".

[create a47r3]
a47r3 is a area. "An area (47) in r3".
the printed name of the a47r3 is "Room 3".
Understand "Area 47 in Room 3" as a47r3.
the parent of the a47r3 is r3.
the X of the a47r3 is "549538.5509290386".
the Y of the a47r3 is "1036261.0062088706".

[create a49r3]
a49r3 is a area. "An area (49) in r3".
the printed name of the a49r3 is "Room 3".
Understand "Area 49 in Room 3" as a49r3.
the parent of the a49r3 is r3.
the X of the a49r3 is "548874.4921500897".
the Y of the a49r3 is "1036260.7986347189".

[create a61r3]
a61r3 is a area. "An area (61) in r3".
the printed name of the a61r3 is "Room 3".
Understand "Area 61 in Room 3" as a61r3.
the parent of the a61r3 is r3.
the X of the a61r3 is "549208.9486065779".
the Y of the a61r3 is "1036178.1064398694".

[create a0r4]
a0r4 is a area. "An area (0) in r4".
the printed name of the a0r4 is "Room 4".
Understand "Area 0 in Room 4" as a0r4.
the parent of the a0r4 is r4.
the X of the a0r4 is "547358.0".
the Y of the a0r4 is "1036314.0".

[create a1r4]
a1r4 is a area. "An area (1) in r4".
the printed name of the a1r4 is "Room 4".
Understand "Area 1 in Room 4" as a1r4.
the parent of the a1r4 is r4.
the X of the a1r4 is "547327.1050372068".
the Y of the a1r4 is "1036287.7894485784".

[create a6r4]
a6r4 is a area. "An area (6) in r4".
the printed name of the a6r4 is "Room 4".
Understand "Area 6 in Room 4" as a6r4.
the parent of the a6r4 is r4.
the X of the a6r4 is "546535.0".
the Y of the a6r4 is "1036314.0".

[create a7r4]
a7r4 is a area. "An area (7) in r4".
the printed name of the a7r4 is "Room 4".
Understand "Area 7 in Room 4" as a7r4.
the parent of the a7r4 is r4.
the X of the a7r4 is "546566.0876068377".
the Y of the a7r4 is "1036288.1662393161".

[create a16r4]
a16r4 is a area. "An area (16) in r4".
the printed name of the a16r4 is "Room 4".
Understand "Area 16 in Room 4" as a16r4.
the parent of the a16r4 is r4.
the X of the a16r4 is "546526.0".
the Y of the a16r4 is "1036224.0".

[create a17r4]
a17r4 is a area. "An area (17) in r4".
the printed name of the a17r4 is "Room 4".
Understand "Area 17 in Room 4" as a17r4.
the parent of the a17r4 is r4.
the X of the a17r4 is "546565.4941364605".
the Y of the a17r4 is "1036242.7755863538".

[create a18r4]
a18r4 is a area. "An area (18) in r4".
the printed name of the a18r4 is "Room 4".
Understand "Area 18 in Room 4" as a18r4.
the parent of the a18r4 is r4.
the X of the a18r4 is "547368.0".
the Y of the a18r4 is "1036224.0".

[create a19r4]
a19r4 is a area. "An area (19) in r4".
the printed name of the a19r4 is "Room 4".
Understand "Area 19 in Room 4" as a19r4.
the parent of the a19r4 is r4.
the X of the a19r4 is "547327.6686930092".
the Y of the a19r4 is "1036242.2965214454".

[create a48r4]
a48r4 is a area. "An area (48) in r4".
the printed name of the a48r4 is "Room 4".
Understand "Area 48 in Room 4" as a48r4.
the parent of the a48r4 is r4.
the X of the a48r4 is "546582.2813718212".
the Y of the a48r4 is "1036261.2845894439".

[create a50r4]
a50r4 is a area. "An area (50) in r4".
the printed name of the a50r4 is "Room 4".
Understand "Area 50 in Room 4" as a50r4.
the parent of the a50r4 is r4.
the X of the a50r4 is "547311.1653756483".
the Y of the a50r4 is "1036261.2658517451".

[create a66r4]
a66r4 is a area. "An area (66) in r4".
the printed name of the a66r4 is "Room 4".
Understand "Area 66 in Room 4" as a66r4.
the parent of the a66r4 is r4.
the X of the a66r4 is "546937.79770735".
the Y of the a66r4 is "1036144.5677680377".

[create a8r5]
a8r5 is a area. "An area (8) in r5".
the printed name of the a8r5 is "Room 5".
Understand "Area 8 in Room 5" as a8r5.
the parent of the a8r5 is r5.
the X of the a8r5 is "545777.0".
the Y of the a8r5 is "1034280.0".

[create a9r5]
a9r5 is a area. "An area (9) in r5".
the printed name of the a9r5 is "Room 5".
Understand "Area 9 in Room 5" as a9r5.
the parent of the a9r5 is r5.
the X of the a9r5 is "545736.7054722287".
the Y of the a9r5 is "1034261.5971994376".

[create a10r5]
a10r5 is a area. "An area (10) in r5".
the printed name of the a10r5 is "Room 5".
Understand "Area 10 in Room 5" as a10r5.
the parent of the a10r5 is r5.
the X of the a10r5 is "546272.0".
the Y of the a10r5 is "1034280.0".

[create a11r5]
a11r5 is a area. "An area (11) in r5".
the printed name of the a11r5 is "Room 5".
Understand "Area 11 in Room 5" as a11r5.
the parent of the a11r5 is r5.
the X of the a11r5 is "546313.3556858187".
the Y of the a11r5 is "1034261.7045075546".

[create a25r5]
a25r5 is a area. "An area (25) in r5".
the printed name of the a25r5 is "Room 5".
Understand "Area 25 in Room 5" as a25r5.
the parent of the a25r5 is r5.
the X of the a25r5 is "547091.7862838916".
the Y of the a25r5 is "1034216.6339712921".

[create a26r5]
a26r5 is a area. "An area (26) in r5".
the printed name of the a26r5 is "Room 5".
Understand "Area 26 in Room 5" as a26r5.
the parent of the a26r5 is r5.
the X of the a26r5 is "545009.0".
the Y of the a26r5 is "1034191.0".

[create a27r5]
a27r5 is a area. "An area (27) in r5".
the printed name of the a27r5 is "Room 5".
Understand "Area 27 in Room 5" as a27r5.
the parent of the a27r5 is r5.
the X of the a27r5 is "545040.2137161086".
the Y of the a27r5 is "1034216.633971292".

[create a28r5]
a28r5 is a area. "An area (28) in r5".
the printed name of the a28r5 is "Room 5".
Understand "Area 28 in Room 5" as a28r5.
the parent of the a28r5 is r5.
the X of the a28r5 is "545768.0".
the Y of the a28r5 is "1034191.0".

[create a29r5]
a29r5 is a area. "An area (29) in r5".
the printed name of the a29r5 is "Room 5".
Understand "Area 29 in Room 5" as a29r5.
the parent of the a29r5 is r5.
the X of the a29r5 is "545736.7862838916".
the Y of the a29r5 is "1034216.6339712921".

[create a30r5]
a30r5 is a area. "An area (30) in r5".
the printed name of the a30r5 is "Room 5".
Understand "Area 30 in Room 5" as a30r5.
the parent of the a30r5 is r5.
the X of the a30r5 is "546281.0".
the Y of the a30r5 is "1034191.0".

[create a31r5]
a31r5 is a area. "An area (31) in r5".
the printed name of the a31r5 is "Room 5".
Understand "Area 31 in Room 5" as a31r5.
the parent of the a31r5 is r5.
the X of the a31r5 is "546312.2137161084".
the Y of the a31r5 is "1034216.6339712921".

[create a32r5]
a32r5 is a area. "An area (32) in r5".
the printed name of the a32r5 is "Room 5".
Understand "Area 32 in Room 5" as a32r5.
the parent of the a32r5 is r5.
the X of the a32r5 is "547132.0".
the Y of the a32r5 is "1034280.0".

[create a33r5]
a33r5 is a area. "An area (33) in r5".
the printed name of the a33r5 is "Room 5".
Understand "Area 33 in Room 5" as a33r5.
the parent of the a33r5 is r5.
the X of the a33r5 is "547092.3329344183".
the Y of the a33r5 is "1034261.533748205".

[create a34r5]
a34r5 is a area. "An area (34) in r5".
the printed name of the a34r5 is "Room 5".
Understand "Area 34 in Room 5" as a34r5.
the parent of the a34r5 is r5.
the X of the a34r5 is "545000.0".
the Y of the a34r5 is "1034280.0".

[create a35r5]
a35r5 is a area. "An area (35) in r5".
the printed name of the a35r5 is "Room 5".
Understand "Area 35 in Room 5" as a35r5.
the parent of the a35r5 is r5.
the X of the a35r5 is "545040.2945277713".
the Y of the a35r5 is "1034261.5971994374".

[create a53r5]
a53r5 is a area. "An area (53) in r5".
the printed name of the a53r5 is "Room 5".
Understand "Area 53 in Room 5" as a53r5.
the parent of the a53r5 is r5.
the X of the a53r5 is "545720.6659021408".
the Y of the a53r5 is "1034243.5012742099".

[create a54r5]
a54r5 is a area. "An area (54) in r5".
the printed name of the a54r5 is "Room 5".
Understand "Area 54 in Room 5" as a54r5.
the parent of the a54r5 is r5.
the X of the a54r5 is "546328.634469697".
the Y of the a54r5 is "1034244.0018939393".

[create a55r5]
a55r5 is a area. "An area (55) in r5".
the printed name of the a55r5 is "Room 5".
Understand "Area 55 in Room 5" as a55r5.
the parent of the a55r5 is r5.
the X of the a55r5 is "547075.7901234569".
the Y of the a55r5 is "1034243.2942386833".

[create a56r5]
a56r5 is a area. "An area (56) in r5".
the printed name of the a56r5 is "Room 5".
Understand "Area 56 in Room 5" as a56r5.
the parent of the a56r5 is r5.
the X of the a56r5 is "545056.3340978592".
the Y of the a56r5 is "1034243.5012742099".

[create a64r5]
a64r5 is a area. "An area (64) in r5".
the printed name of the a64r5 is "Room 5".
Understand "Area 64 in Room 5" as a64r5.
the parent of the a64r5 is r5.
the X of the a64r5 is "545604.9376088154".
the Y of the a64r5 is "1035042.9379071898".

[create a65r5]
a65r5 is a area. "An area (65) in r5".
the printed name of the a65r5 is "Room 5".
Understand "Area 65 in Room 5" as a65r5.
the parent of the a65r5 is r5.
the X of the a65r5 is "546472.9076679606".
the Y of the a65r5 is "1035056.9294065497".

[create a67r5]
a67r5 is a area. "An area (67) in r5".
the printed name of the a67r5 is "Room 5".
Understand "Area 67 in Room 5" as a67r5.
the parent of the a67r5 is r5.
the X of the a67r5 is "545383.3961068324".
the Y of the a67r5 is "1034328.7338161521".

[create a68r5]
a68r5 is a area. "An area (68) in r5".
the printed name of the a68r5 is "Room 5".
Understand "Area 68 in Room 5" as a68r5.
the parent of the a68r5 is r5.
the X of the a68r5 is "546709.4310969368".
the Y of the a68r5 is "1034367.4370029137".

[create a22r6]
a22r6 is a area. "An area (22) in r6".
the printed name of the a22r6 is "Room 6".
Understand "Area 22 in Room 6" as a22r6.
the parent of the a22r6 is r6.
the X of the a22r6 is "545888.0".
the Y of the a22r6 is "1036301.0".

[create a23r6]
a23r6 is a area. "An area (23) in r6".
the printed name of the a23r6 is "Room 6".
Understand "Area 23 in Room 6" as a23r6.
the parent of the a23r6 is r6.
the X of the a23r6 is "545928.1595627209".
the Y of the a23r6 is "1036246.4402178593".

[create a42r6]
a42r6 is a area. "An area (42) in r6".
the printed name of the a42r6 is "Room 6".
Understand "Area 42 in Room 6" as a42r6.
the parent of the a42r6 is r6.
the X of the a42r6 is "546155.0".
the Y of the a42r6 is "1036332.0".

[create a43r6]
a43r6 is a area. "An area (43) in r6".
the printed name of the a43r6 is "Room 6".
Understand "Area 43 in Room 6" as a43r6.
the parent of the a43r6 is r6.
the X of the a43r6 is "546127.259315724".
the Y of the a43r6 is "1036279.5971839246".

[create a52r6]
a52r6 is a area. "An area (52) in r6".
the printed name of the a52r6 is "Room 6".
Understand "Area 52 in Room 6" as a52r6.
the parent of the a52r6 is r6.
the X of the a52r6 is "546002.5871593202".
the Y of the a52r6 is "1036164.2265159852".

[create a62r6]
a62r6 is a area. "An area (62) in r6".
the printed name of the a62r6 is "Room 6".
Understand "Area 62 in Room 6" as a62r6.
the parent of the a62r6 is r6.
the X of the a62r6 is "546054.7557274038".
the Y of the a62r6 is "1036163.3385288817".

[create d0]
d0 is a door. "A door between a58r2 and a61r3".
d0 is north of a58r2 and south of a61r3.

[create d2]
d2 is a door. "A door between a60r0 and a63r1".
d2 is north of a60r0 and south of a63r1.

[create d4]
d4 is a door. "A door between a59r0 and a66r4".
d4 is north of a59r0 and south of a66r4.

[create d6]
d6 is a door. "A door between a63r1 and a57r2".
d6 is southeast of a63r1 and northwest of a57r2.

[create d8]
d8 is a door. "A door between a65r5 and a62r6".
d8 is north of a65r5 and south of a62r6.

[create d9]
d9 is a door. "A door between a64r5 and a52r6".
d9 is north of a64r5 and south of a52r6.

southwest of a0r4 is northeast of a1r4.

southeast of a2r3 is northwest of a3r3.

southwest of a4r3 is northeast of a5r3.

southeast of a6r4 is northwest of a7r4.

southwest of a8r5 is northeast of a9r5.

southeast of a10r5 is northwest of a11r5.

northeast of a12r3 is southwest of a13r3.

northwest of a14r3 is southeast of a15r3.

northeast of a16r4 is southwest of a17r4.

northwest of a18r4 is southeast of a19r4.

southwest of a20r1 is northeast of a21r1.

southeast of a22r6 is northwest of a23r6.

northeast of a26r5 is southwest of a27r5.

northwest of a28r5 is southeast of a29r5.

northeast of a30r5 is southwest of a31r5.

southwest of a32r5 is northeast of a33r5.

southeast of a34r5 is northwest of a35r5.

northwest of a36r0 is southeast of a37r0.

northwest of a38r2 is southeast of a39r2.

southeast of a40r1 is northwest of a41r1.

southwest of a42r6 is northeast of a43r6.

northeast of a24r0 is southwest of a44r0.

northeast of a45r2 is southwest of a46r2.

southwest of a5r3 is northeast of a47r3.

southeast of a7r4 is northwest of a48r4.

northeast of a13r3 is southwest of a49r3.

northwest of a15r3 is southeast of a47r3.

northeast of a17r4 is southwest of a48r4.

northwest of a19r4 is southeast of a50r4.

southwest of a21r1 is northeast of a51r1.

southeast of a23r6 is northwest of a52r6.

southwest of a1r4 is northeast of a50r4.

southeast of a3r3 is northwest of a49r3.

northwest of a29r5 is southeast of a53r5.

northeast of a31r5 is southwest of a54r5.

southwest of a33r5 is northeast of a55r5.

southeast of a35r5 is northwest of a56r5.

southwest of a9r5 is northeast of a53r5.

southeast of a11r5 is northwest of a54r5.

east of a57r2 is west of a58r2.

east of a59r0 is west of a60r0.

northwest of a25r5 is southeast of a55r5.

northeast of a27r5 is southwest of a56r5.

east of a49r3 is west of a61r3.

west of a47r3 is east of a61r3.

east of a52r6 is west of a62r6.

southwest of a43r6 is northeast of a62r6.

east of a64r5 is west of a65r5.

south of a51r1 is north of a63r1.

southeast of a41r1 is northwest of a51r1.

east of a48r4 is west of a66r4.

west of a50r4 is east of a66r4.

northeast of a46r2 is southwest of a57r2.

south of a64r5 is north of a67r5.

northwest of a37r0 is southeast of a60r0.

northwest of a39r2 is southeast of a58r2.

south of a65r5 is north of a68r5.

east of a54r5 is west of a68r5.

east of a56r5 is west of a67r5.

west of a53r5 is east of a67r5.

west of a55r5 is east of a68r5.

northeast of a44r0 is southwest of a59r0.

Definition: a direction (called thatway) is viable if the room thatway from the location is not nowhere.

nLooking is a number that varies. nLooking is 0. 
dirNumber is a number that varies.
dirNumber is 0. 
relDirDesc is a text that varies. 

[describe areas and rooms]
After looking:
	now nLooking is 1; 
	let accessibleRooms be a list of rooms;
	let accessibleAreas be a list of areas;
	let pDirections be list of viable directions;
	let parentSource be the parent of the location of player;
	let relDirs be a list of number;
	repeat with dirToLookAt running through pDirections:
		try silently going dirToLookAt;
		if rule succeeded:
			now dirNumber is 0;
			if "[dirToLookAt]" is "south":
				now dirNumber is 4;
			if "[dirToLookAt]" is "east":
				now dirNumber is 6;
			if "[dirToLookAt]" is "west":
				now dirNumber is 2;
			if "[dirToLookAt]" is "northwest":
				now dirNumber is 1;
			if "[dirToLookAt]" is "southwest":
				now dirNumber is 3;
			if "[dirToLookAt]" is "southeast":
				now dirNumber is 5;
			if "[dirToLookAt]" is "northeast":
				now dirNumber is 7;
			let relDir be the remainder after dividing the orientation of the player - dirNumber + 80 by 8;
			add relDir to relDirs;
			if relDir is 0:
				now relDirDesc is "at the front";
			if relDir is 1:
				now relDirDesc is "at the slight right";
			if relDir is 2:
				now relDirDesc is "at the right";
			if relDir is 3:
				now relDirDesc is "at the sharp right";
			if relDir is 4:
				now relDirDesc is "at the back";
			if relDir is 5:
				now relDirDesc is "at the sharp left";
			if relDir is 6:
				now relDirDesc is "at the left";
			if relDir is 7:
				now relDirDesc is "at the slight left";
			let destinationParent be the parent of the location of the player;
			if "[parentSource]" is "[destinationParent]":
				say "You can continue in the [parentSource] by going [dirToLookAt] ([relDirDesc])[line break]";
				add the location of the player to accessibleAreas;
			otherwise:
				say "You can enter the [destinationParent] by going [dirToLookAt] ([relDirDesc])[line break]";
				add the destinationParent to accessibleRooms;
			try silently going the opposite of dirToLookAt;
	repeat with vO running through the visible_objects of the location:
		say "[vO] is visible from here, but too far! You can move in this room to examine or access [it]";
	now nLooking is 0.
Before going through a locked door when nLooking is 1:
	stop the action.
Going front is an action applying to nothing. Understand "go front" as going front.

Check going front:
	if the orientation of the player is:
		-- 0: try the player going north;
		-- 1: try the player going northwest;
		-- 2: try the player going west;
		-- 3: try the player going southwest;
		-- 4: try the player going south;
		-- 5: try the player going southeast;
		-- 6: try the player going east;
		-- 7: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".


Going back is an action applying to nothing. Understand "go back" as going back.

Check going back:
	if the orientation of the player is:
		-- 4: try the player going north;
		-- 5: try the player going northwest;
		-- 6: try the player going west;
		-- 7: try the player going southwest;
		-- 0: try the player going south;
		-- 1: try the player going southeast;
		-- 2: try the player going east;
		-- 3: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".



Going left is an action applying to nothing. Understand "go left" as going left.

Check going left:
	if the orientation of the player is:
		-- 6: try the player going north;
		-- 7: try the player going northwest;
		-- 0: try the player going west;
		-- 1: try the player going southwest;
		-- 2: try the player going south;
		-- 3: try the player going southeast;
		-- 4: try the player going east;
		-- 5: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".


Going right is an action applying to nothing. Understand "go right" as going right.

Check going right:
	if the orientation of the player is:
		-- 2: try the player going north;
		-- 3: try the player going northwest;
		-- 4: try the player going west;
		-- 5: try the player going southwest;
		-- 6: try the player going south;
		-- 7: try the player going southeast;
		-- 0: try the player going east;
		-- 1: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".

Going sharp right is an action applying to nothing. Understand "go sharp right" as going sharp right.
Going slight right is an action applying to nothing. Understand "go slight right" as going slight right.
Going sharp left is an action applying to nothing. Understand "go sharp left" as going sharp left.
Going slight left is an action applying to nothing. Understand "go slight left" as going slight left.

Check going sharp right:
	if the orientation of the player is:
		-- 3: try the player going north;
		-- 4: try the player going northwest;
		-- 5: try the player going west;
		-- 6: try the player going southwest;
		-- 7: try the player going south;
		-- 0: try the player going southeast;
		-- 1: try the player going east;
		-- 2: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".

Check going slight right:
	if the orientation of the player is:
		-- 1: try the player going north;
		-- 2: try the player going northwest;
		-- 3: try the player going west;
		-- 4: try the player going southwest;
		-- 5: try the player going south;
		-- 6: try the player going southeast;
		-- 7: try the player going east;
		-- 0: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".

Check going sharp left:
	if the orientation of the player is:
		-- 5: try the player going north;
		-- 6: try the player going northwest;
		-- 7: try the player going west;
		-- 0: try the player going southwest;
		-- 1: try the player going south;
		-- 2: try the player going southeast;
		-- 3: try the player going east;
		-- 4: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".

Check going slight left:
	if the orientation of the player is:
		-- 7: try the player going north;
		-- 0: try the player going northwest;
		-- 1: try the player going west;
		-- 2: try the player going southwest;
		-- 3: try the player going south;
		-- 4: try the player going southeast;
		-- 5: try the player going east;
		-- 6: try the player going northeast;
		-- otherwise: say "Yaaaaaa Babaaaaam!!!".


Instead of going north:
	if nLooking is 0:
		now the orientation of the player is 0;
	continue the action.

Instead of going south:
	if nLooking is 0:
		now the orientation of the player is 4;
	continue the action.

Instead of going east:
	if nLooking is 0:
		now the orientation of the player is 6;
	continue the action.

Instead of going west:
	if nLooking is 0:
		now the orientation of the player is 2;
	continue the action.

Instead of going northwest:
	if nLooking is 0:
		now the orientation of the player is 1;
	continue the action.

Instead of going southwest:
	if nLooking is 0:
		now the orientation of the player is 3;
	continue the action.

Instead of going northeast:
	if nLooking is 0:
		now the orientation of the player is 7;
	continue the action.

Instead of going southeast:
	if nLooking is 0:
		now the orientation of the player is 5;
	continue the action.

Understand "veer left" as going slight left.
Understand "veer right" as going slight right.
Understand "turn left" as going left.
Understand "turn right" as going right.
Understand "turn around" as going back.
Understand "turn sharp left" as going sharp left.
Understand "Go straight" as going front.
Understand "turn sharp right" as going sharp right.
Understand "Turn slightly left" as going slight left.
Understand "Turn slightly right" as going slight right.
Understand "Make a sharp right" as going sharp right.
Understand "Make a sharp left" as going sharp left.
Every turn:
	let xx be the X of the location of the player;
	let yy be the Y of the location of the player;
	say "X: [xx][line break]";
	say "Y: [yy]".
the player is in a6r4.

the orientation of the player is 6.
