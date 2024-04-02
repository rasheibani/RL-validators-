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

[create a13r0]
a13r0 is a area. "An area (13) in r0".
the printed name of the a13r0 is "Room 0".
Understand "Area 13 in Room 0" as a13r0.
the parent of the a13r0 is r0.
the X of the a13r0 is "545890.0086022713".
the Y of the a13r0 is "1672764.0104014394".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "545786.8816727516".
the Y of the a23r0 is "1672619.495295859".

[create a25r0]
a25r0 is a area. "An area (25) in r0".
the printed name of the a25r0 is "Room 0".
Understand "Area 25 in Room 0" as a25r0.
the parent of the a25r0 is r0.
the X of the a25r0 is "545903.7496886398".
the Y of the a25r0 is "1672762.3173736057".

[create a26r0]
a26r0 is a area. "An area (26) in r0".
the printed name of the a26r0 is "Room 0".
Understand "Area 26 in Room 0" as a26r0.
the parent of the a26r0 is r0.
the X of the a26r0 is "545997.2968536066".
the Y of the a26r0 is "1672651.4209743903".

[create a67r0]
a67r0 is a area. "An area (67) in r0".
the printed name of the a67r0 is "Room 0".
Understand "Area 67 in Room 0" as a67r0.
the parent of the a67r0 is r0.
the X of the a67r0 is "545756.0".
the Y of the a67r0 is "1672572.0".

[create a69r0]
a69r0 is a area. "An area (69) in r0".
the printed name of the a69r0 is "Room 0".
Understand "Area 69 in Room 0" as a69r0.
the parent of the a69r0 is r0.
the X of the a69r0 is "546038.0".
the Y of the a69r0 is "1672600.0".

[create a0r1]
a0r1 is a area. "An area (0) in r1".
the printed name of the a0r1 is "Room 1".
Understand "Area 0 in Room 1" as a0r1.
the parent of the a0r1 is r1.
the X of the a0r1 is "546544.5167685037".
the Y of the a0r1 is "1674573.513395759".

[create a1r1]
a1r1 is a area. "An area (1) in r1".
the printed name of the a1r1 is "Room 1".
Understand "Area 1 in Room 1" as a1r1.
the parent of the a1r1 is r1.
the X of the a1r1 is "546549.8708448356".
the Y of the a1r1 is "1674405.3162913686".

[create a4r1]
a4r1 is a area. "An area (4) in r1".
the printed name of the a4r1 is "Room 1".
Understand "Area 4 in Room 1" as a4r1.
the parent of the a4r1 is r1.
the X of the a4r1 is "546477.3333333333".
the Y of the a4r1 is "1674660.9694444442".

[create a53r1]
a53r1 is a area. "An area (53) in r1".
the printed name of the a53r1 is "Room 1".
Understand "Area 53 in Room 1" as a53r1.
the parent of the a53r1 is r1.
the X of the a53r1 is "546644.0".
the Y of the a53r1 is "1674704.0".

[create a54r1]
a54r1 is a area. "An area (54) in r1".
the printed name of the a54r1 is "Room 1".
Understand "Area 54 in Room 1" as a54r1.
the parent of the a54r1 is r1.
the X of the a54r1 is "546610.6666666666".
the Y of the a54r1 is "1674653.5364495008".

[create a71r1]
a71r1 is a area. "An area (71) in r1".
the printed name of the a71r1 is "Room 1".
Understand "Area 71 in Room 1" as a71r1.
the parent of the a71r1 is r1.
the X of the a71r1 is "546444.0".
the Y of the a71r1 is "1674704.0".

[create a14r2]
a14r2 is a area. "An area (14) in r2".
the printed name of the a14r2 is "Room 2".
Understand "Area 14 in Room 2" as a14r2.
the parent of the a14r2 is r2.
the X of the a14r2 is "547144.2390740267".
the Y of the a14r2 is "1672762.431282769".

[create a24r2]
a24r2 is a area. "An area (24) in r2".
the printed name of the a24r2 is "Room 2".
Understand "Area 24 in Room 2" as a24r2.
the parent of the a24r2 is r2.
the X of the a24r2 is "547048.7043713275".
the Y of the a24r2 is "1672622.7067112252".

[create a27r2]
a27r2 is a area. "An area (27) in r2".
the printed name of the a27r2 is "Room 2".
Understand "Area 27 in Room 2" as a27r2.
the parent of the a27r2 is r2.
the X of the a27r2 is "547166.2896516462".
the Y of the a27r2 is "1672760.8833195833".

[create a28r2]
a28r2 is a area. "An area (28) in r2".
the printed name of the a28r2 is "Room 2".
Understand "Area 28 in Room 2" as a28r2.
the parent of the a28r2 is r2.
the X of the a28r2 is "547260.4539705468".
the Y of the a28r2 is "1672648.7643334027".

[create a68r2]
a68r2 is a area. "An area (68) in r2".
the printed name of the a68r2 is "Room 2".
Understand "Area 68 in Room 2" as a68r2.
the parent of the a68r2 is r2.
the X of the a68r2 is "547018.0".
the Y of the a68r2 is "1672572.0".

[create a70r2]
a70r2 is a area. "An area (70) in r2".
the printed name of the a70r2 is "Room 2".
Understand "Area 70 in Room 2" as a70r2.
the parent of the a70r2 is r2.
the X of the a70r2 is "547301.0".
the Y of the a70r2 is "1672600.0".

[create a2r3]
a2r3 is a area. "An area (2) in r3".
the printed name of the a2r3 is "Room 3".
Understand "Area 2 in Room 3" as a2r3.
the parent of the a2r3 is r3.
the X of the a2r3 is "547342.4921500897".
the Y of the a2r3 is "1674660.7986347186".

[create a3r3]
a3r3 is a area. "An area (3) in r3".
the printed name of the a3r3 is "Room 3".
Understand "Area 3 in Room 3" as a3r3.
the parent of the a3r3 is r3.
the X of the a3r3 is "547630.4944995979".
the Y of the a3r3 is "1674611.0205706104".

[create a5r3]
a5r3 is a area. "An area (5) in r3".
the printed name of the a5r3 is "Room 3".
Understand "Area 5 in Room 3" as a5r3.
the parent of the a5r3 is r3.
the X of the a5r3 is "548006.5675408735".
the Y of the a5r3 is "1674553.6819442695".

[create a6r3]
a6r3 is a area. "An area (6) in r3".
the printed name of the a6r3 is "Room 3".
Understand "Area 6 in Room 3" as a6r3.
the parent of the a6r3 is r3.
the X of the a6r3 is "548939.626695049".
the Y of the a6r3 is "1674586.6535457706".

[create a10r3]
a10r3 is a area. "An area (10) in r3".
the printed name of the a10r3 is "Room 3".
Understand "Area 10 in Room 3" as a10r3.
the parent of the a10r3 is r3.
the X of the a10r3 is "547677.1587808633".
the Y of the a10r3 is "1674581.377277184".

[create a11r3]
a11r3 is a area. "An area (11) in r3".
the printed name of the a11r3 is "Room 3".
Understand "Area 11 in Room 3" as a11r3.
the parent of the a11r3 is r3.
the X of the a11r3 is "547994.2203248537".
the Y of the a11r3 is "1674541.680266012".

[create a12r3]
a12r3 is a area. "An area (12) in r3".
the printed name of the a12r3 is "Room 3".
Understand "Area 12 in Room 3" as a12r3.
the parent of the a12r3 is r3.
the X of the a12r3 is "549033.0550595238".
the Y of the a12r3 is "1674700.0".

[create a15r3]
a15r3 is a area. "An area (15) in r3".
the printed name of the a15r3 is "Room 3".
Understand "Area 15 in Room 3" as a15r3.
the parent of the a15r3 is r3.
the X of the a15r3 is "547999.0929783394".
the Y of the a15r3 is "1673666.0".

[create a17r3]
a17r3 is a area. "An area (17) in r3".
the printed name of the a17r3 is "Room 3".
Understand "Area 17 in Room 3" as a17r3.
the parent of the a17r3 is r3.
the X of the a17r3 is "548756.7255000001".
the Y of the a17r3 is "1673666.0".

[create a19r3]
a19r3 is a area. "An area (19) in r3".
the printed name of the a19r3 is "Room 3".
Understand "Area 19 in Room 3" as a19r3.
the parent of the a19r3 is r3.
the X of the a19r3 is "548825.5748885327".
the Y of the a19r3 is "1673961.273880704".

[create a20r3]
a20r3 is a area. "An area (20) in r3".
the printed name of the a20r3 is "Room 3".
Understand "Area 20 in Room 3" as a20r3.
the parent of the a20r3 is r3.
the X of the a20r3 is "549049.25246563".
the Y of the a20r3 is "1674258.466890702".

[create a31r3]
a31r3 is a area. "An area (31) in r3".
the printed name of the a31r3 is "Room 3".
Understand "Area 31 in Room 3" as a31r3.
the parent of the a31r3 is r3.
the X of the a31r3 is "547295.0".
the Y of the a31r3 is "1674714.0".

[create a32r3]
a32r3 is a area. "An area (32) in r3".
the printed name of the a32r3 is "Room 3".
Understand "Area 32 in Room 3" as a32r3.
the parent of the a32r3 is r3.
the X of the a32r3 is "547325.9153780066".
the Y of the a32r3 is "1674688.1834621993".

[create a35r3]
a35r3 is a area. "An area (35) in r3".
the printed name of the a35r3 is "Room 3".
Understand "Area 35 in Room 3" as a35r3.
the parent of the a35r3 is r3.
the X of the a35r3 is "547589.0".
the Y of the a35r3 is "1674714.0".

[create a36r3]
a36r3 is a area. "An area (36) in r3".
the printed name of the a36r3 is "Room 3".
Understand "Area 36 in Room 3" as a36r3.
the parent of the a36r3 is r3.
the X of the a36r3 is "547620.8264604812".
the Y of the a36r3 is "1674705.767353952".

[create a37r3]
a37r3 is a area. "An area (37) in r3".
the printed name of the a37r3 is "Room 3".
Understand "Area 37 in Room 3" as a37r3.
the parent of the a37r3 is r3.
the X of the a37r3 is "548054.0".
the Y of the a37r3 is "1674714.0".

[create a38r3]
a38r3 is a area. "An area (38) in r3".
the printed name of the a38r3 is "Room 3".
Understand "Area 38 in Room 3" as a38r3.
the parent of the a38r3 is r3.
the X of the a38r3 is "548022.0059829061".
the Y of the a38r3 is "1674705.7505982907".

[create a39r3]
a39r3 is a area. "An area (39) in r3".
the printed name of the a39r3 is "Room 3".
Understand "Area 39 in Room 3" as a39r3.
the parent of the a39r3 is r3.
the X of the a39r3 is "548888.0".
the Y of the a39r3 is "1674010.0".

[create a40r3]
a40r3 is a area. "An area (40) in r3".
the printed name of the a40r3 is "Room 3".
Understand "Area 40 in Room 3" as a40r3.
the parent of the a40r3 is r3.
the X of the a40r3 is "548857.9799058654".
the Y of the a40r3 is "1673983.8416606323".

[create a41r3]
a41r3 is a area. "An area (41) in r3".
the printed name of the a41r3 is "Room 3".
Understand "Area 41 in Room 3" as a41r3.
the parent of the a41r3 is r3.
the X of the a41r3 is "548781.0".
the Y of the a41r3 is "1674022.0".

[create a42r3]
a42r3 is a area. "An area (42) in r3".
the printed name of the a42r3 is "Room 3".
Understand "Area 42 in Room 3" as a42r3.
the parent of the a42r3 is r3.
the X of the a42r3 is "548804.4879350015".
the Y of the a42r3 is "1673989.9132537628".

[create a43r3]
a43r3 is a area. "An area (43) in r3".
the printed name of the a43r3 is "Room 3".
Understand "Area 43 in Room 3" as a43r3.
the parent of the a43r3 is r3.
the X of the a43r3 is "549017.0".
the Y of the a43r3 is "1674188.0".

[create a44r3]
a44r3 is a area. "An area (44) in r3".
the printed name of the a44r3 is "Room 3".
Understand "Area 44 in Room 3" as a44r3.
the parent of the a44r3 is r3.
the X of the a44r3 is "549025.2487853912".
the Y of the a44r3 is "1674225.2932652037".

[create a45r3]
a45r3 is a area. "An area (45) in r3".
the printed name of the a45r3 is "Room 3".
Understand "Area 45 in Room 3" as a45r3.
the parent of the a45r3 is r3.
the X of the a45r3 is "549100.0".
the Y of the a45r3 is "1674225.0".

[create a46r3]
a46r3 is a area. "An area (46) in r3".
the printed name of the a46r3 is "Room 3".
Understand "Area 46 in Room 3" as a46r3.
the parent of the a46r3 is r3.
the X of the a46r3 is "549062.1025414976".
the Y of the a46r3 is "1674254.2159204248".

[create a47r3]
a47r3 is a area. "An area (47) in r3".
the printed name of the a47r3 is "Room 3".
Understand "Area 47 in Room 3" as a47r3.
the parent of the a47r3 is r3.
the X of the a47r3 is "547286.0".
the Y of the a47r3 is "1674624.0".

[create a48r3]
a48r3 is a area. "An area (48) in r3".
the printed name of the a48r3 is "Room 3".
Understand "Area 48 in Room 3" as a48r3.
the parent of the a48r3 is r3.
the X of the a48r3 is "547326.7622699386".
the Y of the a48r3 is "1674642.6487730062".

[create a72r3]
a72r3 is a area. "An area (72) in r3".
the printed name of the a72r3 is "Room 3".
Understand "Area 72 in Room 3" as a72r3.
the parent of the a72r3 is r3.
the X of the a72r3 is "549063.0".
the Y of the a72r3 is "1674723.0".

[create a16r4]
a16r4 is a area. "An area (16) in r4".
the printed name of the a16r4 is "Room 4".
Understand "Area 16 in Room 4" as a16r4.
the parent of the a16r4 is r4.
the X of the a16r4 is "547994.0".
the Y of the a16r4 is "1672768.577830189".

[create a18r4]
a18r4 is a area. "An area (18) in r4".
the printed name of the a18r4 is "Room 4".
Understand "Area 18 in Room 4" as a18r4.
the parent of the a18r4 is r4.
the X of the a18r4 is "548825.6508716345".
the Y of the a18r4 is "1673372.6121508037".

[create a21r4]
a21r4 is a area. "An area (21) in r4".
the printed name of the a21r4 is "Room 4".
Understand "Area 21 in Room 4" as a21r4.
the parent of the a21r4 is r4.
the X of the a21r4 is "548360.781769306".
the Y of the a21r4 is "1672644.9748289345".

[create a22r4]
a22r4 is a area. "An area (22) in r4".
the printed name of the a22r4 is "Room 4".
Understand "Area 22 in Room 4" as a22r4.
the parent of the a22r4 is r4.
the X of the a22r4 is "547612.389199255".
the Y of the a22r4 is "1672640.2597765361".

[create a55r4]
a55r4 is a area. "An area (55) in r4".
the printed name of the a55r4 is "Room 4".
Understand "Area 55 in Room 4" as a55r4.
the parent of the a55r4 is r4.
the X of the a55r4 is "547567.0".
the Y of the a55r4 is "1672591.0".

[create a56r4]
a56r4 is a area. "An area (56) in r4".
the printed name of the a56r4 is "Room 4".
Understand "Area 56 in Room 4" as a56r4.
the parent of the a56r4 is r4.
the X of the a56r4 is "547598.2137161085".
the Y of the a56r4 is "1672616.633971292".

[create a57r4]
a57r4 is a area. "An area (57) in r4".
the printed name of the a57r4 is "Room 4".
Understand "Area 57 in Room 4" as a57r4.
the parent of the a57r4 is r4.
the X of the a57r4 is "547558.0".
the Y of the a57r4 is "1672680.0".

[create a58r4]
a58r4 is a area. "An area (58) in r4".
the printed name of the a58r4 is "Room 4".
Understand "Area 58 in Room 4" as a58r4.
the parent of the a58r4 is r4.
the X of the a58r4 is "547589.0059757691".
the Y of the a58r4 is "1672660.657907662".

[create a59r4]
a59r4 is a area. "An area (59) in r4".
the printed name of the a59r4 is "Room 4".
Understand "Area 59 in Room 4" as a59r4.
the parent of the a59r4 is r4.
the X of the a59r4 is "548409.0".
the Y of the a59r4 is "1672591.0".

[create a60r4]
a60r4 is a area. "An area (60) in r4".
the printed name of the a60r4 is "Room 4".
Understand "Area 60 in Room 4" as a60r4.
the parent of the a60r4 is r4.
the X of the a60r4 is "548377.7862838916".
the Y of the a60r4 is "1672616.633971292".

[create a61r4]
a61r4 is a area. "An area (61) in r4".
the printed name of the a61r4 is "Room 4".
Understand "Area 61 in Room 4" as a61r4.
the parent of the a61r4 is r4.
the X of the a61r4 is "548418.0".
the Y of the a61r4 is "1672680.0".

[create a62r4]
a62r4 is a area. "An area (62) in r4".
the printed name of the a62r4 is "Room 4".
Understand "Area 62 in Room 4" as a62r4.
the parent of the a62r4 is r4.
the X of the a62r4 is "548375.1485252037".
the Y of the a62r4 is "1672661.8557671143".

[create a63r4]
a63r4 is a area. "An area (63) in r4".
the printed name of the a63r4 is "Room 4".
Understand "Area 63 in Room 4" as a63r4.
the parent of the a63r4 is r4.
the X of the a63r4 is "548781.0".
the Y of the a63r4 is "1673310.0".

[create a64r4]
a64r4 is a area. "An area (64) in r4".
the printed name of the a64r4 is "Room 4".
Understand "Area 64 in Room 4" as a64r4.
the parent of the a64r4 is r4.
the X of the a64r4 is "548803.796458405".
the Y of the a64r4 is "1673341.9519300456".

[create a65r4]
a65r4 is a area. "An area (65) in r4".
the printed name of the a65r4 is "Room 4".
Understand "Area 65 in Room 4" as a65r4.
the parent of the a65r4 is r4.
the X of the a65r4 is "548888.0".
the Y of the a65r4 is "1673325.0".

[create a66r4]
a66r4 is a area. "An area (66) in r4".
the printed name of the a66r4 is "Room 4".
Understand "Area 66 in Room 4" as a66r4.
the parent of the a66r4 is r4.
the X of the a66r4 is "548856.997262635".
the Y of the a66r4 is "1673351.5861932037".

[create a7r5]
a7r5 is a area. "An area (7) in r5".
the printed name of the a7r5 is "Room 5".
Understand "Area 7 in Room 5" as a7r5.
the parent of the a7r5 is r5.
the X of the a7r5 is "545050.2813718208".
the Y of the a7r5 is "1674661.2845894434".

[create a8r5]
a8r5 is a area. "An area (8) in r5".
the printed name of the a8r5 is "Room 5".
Understand "Area 8 in Room 5" as a8r5.
the parent of the a8r5 is r5.
the X of the a8r5 is "545405.79770735".
the Y of the a8r5 is "1674544.5677680378".

[create a9r5]
a9r5 is a area. "An area (9) in r5".
the printed name of the a9r5 is "Room 5".
Understand "Area 9 in Room 5" as a9r5.
the parent of the a9r5 is r5.
the X of the a9r5 is "545779.1653756485".
the Y of the a9r5 is "1674661.2658517451".

[create a29r5]
a29r5 is a area. "An area (29) in r5".
the printed name of the a29r5 is "Room 5".
Understand "Area 29 in Room 5" as a29r5.
the parent of the a29r5 is r5.
the X of the a29r5 is "545826.0".
the Y of the a29r5 is "1674714.0".

[create a30r5]
a30r5 is a area. "An area (30) in r5".
the printed name of the a30r5 is "Room 5".
Understand "Area 30 in Room 5" as a30r5.
the parent of the a30r5 is r5.
the X of the a30r5 is "545795.1050372067".
the Y of the a30r5 is "1674687.7894485788".

[create a33r5]
a33r5 is a area. "An area (33) in r5".
the printed name of the a33r5 is "Room 5".
Understand "Area 33 in Room 5" as a33r5.
the parent of the a33r5 is r5.
the X of the a33r5 is "545003.0".
the Y of the a33r5 is "1674714.0".

[create a34r5]
a34r5 is a area. "An area (34) in r5".
the printed name of the a34r5 is "Room 5".
Understand "Area 34 in Room 5" as a34r5.
the parent of the a34r5 is r5.
the X of the a34r5 is "545034.0876068375".
the Y of the a34r5 is "1674688.1662393163".

[create a49r5]
a49r5 is a area. "An area (49) in r5".
the printed name of the a49r5 is "Room 5".
Understand "Area 49 in Room 5" as a49r5.
the parent of the a49r5 is r5.
the X of the a49r5 is "544994.0".
the Y of the a49r5 is "1674624.0".

[create a50r5]
a50r5 is a area. "An area (50) in r5".
the printed name of the a50r5 is "Room 5".
Understand "Area 50 in Room 5" as a50r5.
the parent of the a50r5 is r5.
the X of the a50r5 is "545033.4941364606".
the Y of the a50r5 is "1674642.775586354".

[create a51r5]
a51r5 is a area. "An area (51) in r5".
the printed name of the a51r5 is "Room 5".
Understand "Area 51 in Room 5" as a51r5.
the parent of the a51r5 is r5.
the X of the a51r5 is "545836.0".
the Y of the a51r5 is "1674624.0".

[create a52r5]
a52r5 is a area. "An area (52) in r5".
the printed name of the a52r5 is "Room 5".
Understand "Area 52 in Room 5" as a52r5.
the parent of the a52r5 is r5.
the X of the a52r5 is "545795.6686930091".
the Y of the a52r5 is "1674642.2965214455".

[create d0]
d0 is a door. "A door between a27r2 and a10r3".
d0 is north of a27r2 and south of a10r3.

[create d2]
d2 is a door. "A door between a25r0 and a1r1".
d2 is north of a25r0 and south of a1r1.

[create d4]
d4 is a door. "A door between a13r0 and a8r5".
d4 is north of a13r0 and south of a8r5.

[create d6]
d6 is a door. "A door between a1r1 and a14r2".
d6 is southeast of a1r1 and northwest of a14r2.

[create d8]
d8 is a door. "A door between a15r3 and a16r4".
d8 is south of a15r3 and north of a16r4.

[create d9]
d9 is a door. "A door between a17r3 and a18r4".
d9 is south of a17r3 and north of a18r4.

south of a0r1 is north of a1r1.

east of a2r3 is west of a3r3.

northwest of a0r1 is southeast of a4r1.

east of a5r3 is west of a6r3.

east of a7r5 is west of a8r5.

east of a8r5 is west of a9r5.

east of a10r3 is west of a11r3.

northeast of a6r3 is southwest of a12r3.

east of a15r3 is west of a17r3.

north of a17r3 is south of a19r3.

south of a11r3 is north of a15r3.

south of a6r3 is north of a20r3.

east of a16r4 is west of a21r4.

west of a16r4 is east of a22r4.

southwest of a13r0 is northeast of a23r0.

southwest of a14r2 is northeast of a24r2.

southeast of a25r0 is northwest of a26r0.

southeast of a27r2 is northwest of a28r2.

southwest of a29r5 is northeast of a30r5.

southeast of a31r3 is northwest of a32r3.

southeast of a33r5 is northwest of a34r5.

east of a35r3 is west of a36r3.

west of a37r3 is east of a38r3.

southwest of a39r3 is northeast of a40r3.

southeast of a41r3 is northwest of a42r3.

north of a43r3 is south of a44r3.

northwest of a45r3 is southeast of a46r3.

northeast of a47r3 is southwest of a48r3.

northeast of a49r5 is southwest of a50r5.

northwest of a51r5 is southeast of a52r5.

southwest of a53r1 is northeast of a54r1.

northeast of a55r4 is southwest of a56r4.

southeast of a57r4 is northwest of a58r4.

northwest of a59r4 is southeast of a60r4.

southwest of a61r4 is northeast of a62r4.

northeast of a63r4 is southwest of a64r4.

northwest of a65r4 is southeast of a66r4.

southwest of a23r0 is northeast of a67r0.

southwest of a24r2 is northeast of a68r2.

southeast of a26r0 is northwest of a69r0.

southeast of a28r2 is northwest of a70r2.

northwest of a4r1 is southeast of a71r1.

northeast of a12r3 is southwest of a72r3.

northwest of a7r5 is southeast of a34r5.

north of a3r3 is south of a36r3.

north of a5r3 is south of a38r3.

southwest of a5r3 is northeast of a11r3.

southeast of a3r3 is northwest of a10r3.

southwest of a2r3 is northeast of a48r3.

southwest of a7r5 is northeast of a50r5.

southeast of a9r5 is northwest of a52r5.

northeast of a0r1 is southwest of a54r1.

northeast of a9r5 is southwest of a30r5.

northwest of a2r3 is southeast of a32r3.

east of a14r2 is west of a27r2.

east of a13r0 is west of a25r0.

southwest of a18r4 is northeast of a64r4.

southeast of a18r4 is northwest of a66r4.

northeast of a19r3 is southwest of a40r3.

northwest of a19r3 is southeast of a42r3.

southwest of a20r3 is northeast of a44r3.

east of a20r3 is west of a46r3.

southwest of a22r4 is northeast of a56r4.

northwest of a22r4 is southeast of a58r4.

southeast of a21r4 is northwest of a60r4.

northeast of a21r4 is southwest of a62r4.

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
the player is in a12r3.

the orientation of the player is 6.
