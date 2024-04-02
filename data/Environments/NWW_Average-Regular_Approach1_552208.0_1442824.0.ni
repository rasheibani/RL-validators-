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

[create r7]
r7 is a indoor_room. "Room 7".

[create r8]
r8 is a indoor_room. "Room 8".

[create r9]
r9 is a indoor_room. "Room 9".

[create r10]
r10 is a indoor_room. "Room 10".

[create r11]
r11 is a indoor_room. "Room 11".

[create a11r0]
a11r0 is a area. "An area (11) in r0".
the printed name of the a11r0 is "Room 0".
Understand "Area 11 in Room 0" as a11r0.
the parent of the a11r0 is r0.
the X of the a11r0 is "546892.9405397376".
the Y of the a11r0 is "1441020.819968459".

[create a13r0]
a13r0 is a area. "An area (13) in r0".
the printed name of the a13r0 is "Room 0".
Understand "Area 13 in Room 0" as a13r0.
the parent of the a13r0 is r0.
the X of the a13r0 is "546869.6571255578".
the Y of the a13r0 is "1440837.8159683133".

[create a14r0]
a14r0 is a area. "An area (14) in r0".
the printed name of the a14r0 is "Room 0".
Understand "Area 14 in Room 0" as a14r0.
the parent of the a14r0 is r0.
the X of the a14r0 is "546921.9623993273".
the Y of the a14r0 is "1440942.03653799".

[create a26r0]
a26r0 is a area. "An area (26) in r0".
the printed name of the a26r0 is "Room 0".
Understand "Area 26 in Room 0" as a26r0.
the parent of the a26r0 is r0.
the X of the a26r0 is "547057.8020768753".
the Y of the a26r0 is "1440826.7924674517".

[create a102r0]
a102r0 is a area. "An area (102) in r0".
the printed name of the a102r0 is "Room 0".
Understand "Area 102 in Room 0" as a102r0.
the parent of the a102r0 is r0.
the X of the a102r0 is "546846.0".
the Y of the a102r0 is "1440772.0".

[create a109r0]
a109r0 is a area. "An area (109) in r0".
the printed name of the a109r0 is "Room 0".
Understand "Area 109 in Room 0" as a109r0.
the parent of the a109r0 is r0.
the X of the a109r0 is "547092.0".
the Y of the a109r0 is "1440800.0".

[create a8r1]
a8r1 is a area. "An area (8) in r1".
the printed name of the a8r1 is "Room 1".
Understand "Area 8 in Room 1" as a8r1.
the parent of the a8r1 is r1.
the X of the a8r1 is "547227.4760066518".
the Y of the a8r1 is "1442128.6567924877".

[create a12r1]
a12r1 is a area. "An area (12) in r1".
the printed name of the a12r1 is "Room 1".
Understand "Area 12 in Room 1" as a12r1.
the parent of the a12r1 is r1.
the X of the a12r1 is "546965.334830984".
the Y of the a12r1 is "1442120.146706587".

[create a33r1]
a33r1 is a area. "An area (33) in r1".
the printed name of the a33r1 is "Room 1".
Understand "Area 33 in Room 1" as a33r1.
the parent of the a33r1 is r1.
the X of the a33r1 is "546611.2978258749".
the Y of the a33r1 is "1442863.3317228702".

[create a34r1]
a34r1 is a area. "An area (34) in r1".
the printed name of the a34r1 is "Room 1".
Understand "Area 34 in Room 1" as a34r1.
the parent of the a34r1 is r1.
the X of the a34r1 is "546713.1201357462".
the Y of the a34r1 is "1442853.9471220747".

[create a35r1]
a35r1 is a area. "An area (35) in r1".
the printed name of the a35r1 is "Room 1".
Understand "Area 35 in Room 1" as a35r1.
the parent of the a35r1 is r1.
the X of the a35r1 is "547295.3830769429".
the Y of the a35r1 is "1442850.9533877482".

[create a36r1]
a36r1 is a area. "An area (36) in r1".
the printed name of the a36r1 is "Room 1".
Understand "Area 36 in Room 1" as a36r1.
the parent of the a36r1 is r1.
the X of the a36r1 is "547455.9594307821".
the Y of the a36r1 is "1442860.3531446948".

[create a39r1]
a39r1 is a area. "An area (39) in r1".
the printed name of the a39r1 is "Room 1".
Understand "Area 39 in Room 1" as a39r1.
the parent of the a39r1 is r1.
the X of the a39r1 is "547043.1762243432".
the Y of the a39r1 is "1442708.2311254928".

[create a48r1]
a48r1 is a area. "An area (48) in r1".
the printed name of the a48r1 is "Room 1".
Understand "Area 48 in Room 1" as a48r1.
the parent of the a48r1 is r1.
the X of the a48r1 is "547088.1676625136".
the Y of the a48r1 is "1442419.8764480709".

[create a49r1]
a49r1 is a area. "An area (49) in r1".
the printed name of the a49r1 is "Room 1".
Understand "Area 49 in Room 1" as a49r1.
the parent of the a49r1 is r1.
the X of the a49r1 is "547208.5556319004".
the Y of the a49r1 is "1442204.2926013153".

[create a56r1]
a56r1 is a area. "An area (56) in r1".
the printed name of the a56r1 is "Room 1".
Understand "Area 56 in Room 1" as a56r1.
the parent of the a56r1 is r1.
the X of the a56r1 is "546567.0".
the Y of the a56r1 is "1442914.0".

[create a57r1]
a57r1 is a area. "An area (57) in r1".
the printed name of the a57r1 is "Room 1".
Understand "Area 57 in Room 1" as a57r1.
the parent of the a57r1 is r1.
the X of the a57r1 is "546593.6150892761".
the Y of the a57r1 is "1442888.6134910723".

[create a58r1]
a58r1 is a area. "An area (58) in r1".
the printed name of the a58r1 is "Room 1".
Understand "Area 58 in Room 1" as a58r1.
the parent of the a58r1 is r1.
the X of the a58r1 is "547502.0".
the Y of the a58r1 is "1442914.0".

[create a59r1]
a59r1 is a area. "An area (59) in r1".
the printed name of the a59r1 is "Room 1".
Understand "Area 59 in Room 1" as a59r1.
the parent of the a59r1 is r1.
the X of the a59r1 is "547475.0722623172".
the Y of the a59r1 is "1442888.2302513688".

[create a72r1]
a72r1 is a area. "An area (72) in r1".
the printed name of the a72r1 is "Room 1".
Understand "Area 72 in Room 1" as a72r1.
the parent of the a72r1 is r1.
the X of the a72r1 is "547512.0".
the Y of the a72r1 is "1442824.0".

[create a73r1]
a73r1 is a area. "An area (73) in r1".
the printed name of the a73r1 is "Room 1".
Understand "Area 73 in Room 1" as a73r1.
the parent of the a73r1 is r1.
the X of the a73r1 is "547471.6686930091".
the Y of the a73r1 is "1442842.2965214453".

[create a76r1]
a76r1 is a area. "An area (76) in r1".
the printed name of the a76r1 is "Room 1".
Understand "Area 76 in Room 1" as a76r1.
the parent of the a76r1 is r1.
the X of the a76r1 is "546558.0".
the Y of the a76r1 is "1442824.0".

[create a77r1]
a77r1 is a area. "An area (77) in r1".
the printed name of the a77r1 is "Room 1".
Understand "Area 77 in Room 1" as a77r1.
the parent of the a77r1 is r1.
the X of the a77r1 is "546585.2059471365".
the Y of the a77r1 is "1442844.0044052862".

[create a96r1]
a96r1 is a area. "An area (96) in r1".
the printed name of the a96r1 is "Room 1".
Understand "Area 96 in Room 1" as a96r1.
the parent of the a96r1 is r1.
the X of the a96r1 is "547332.0".
the Y of the a96r1 is "1442914.0".

[create a97r1]
a97r1 is a area. "An area (97) in r1".
the printed name of the a97r1 is "Room 1".
Understand "Area 97 in Room 1" as a97r1.
the parent of the a97r1 is r1.
the X of the a97r1 is "547299.973944841".
the Y of the a97r1 is "1442906.3721367521".

[create a98r1]
a98r1 is a area. "An area (98) in r1".
the printed name of the a98r1 is "Room 1".
Understand "Area 98 in Room 1" as a98r1.
the parent of the a98r1 is r1.
the X of the a98r1 is "546678.4226804124".
the Y of the a98r1 is "1442908.2268041237".

[create a99r1]
a99r1 is a area. "An area (99) in r1".
the printed name of the a99r1 is "Room 1".
Understand "Area 99 in Room 1" as a99r1.
the parent of the a99r1 is r1.
the X of the a99r1 is "546711.1034804829".
the Y of the a99r1 is "1442907.874188034".

[create a2r2]
a2r2 is a area. "An area (2) in r2".
the printed name of the a2r2 is "Room 2".
Understand "Area 2 in Room 2" as a2r2.
the parent of the a2r2 is r2.
the X of the a2r2 is "547579.7496886395".
the Y of the a2r2 is "1440962.3173736061".

[create a9r2]
a9r2 is a area. "An area (9) in r2".
the printed name of the a9r2 is "Room 2".
Understand "Area 9 in Room 2" as a9r2.
the parent of the a9r2 is r2.
the X of the a9r2 is "547565.9698254254".
the Y of the a9r2 is "1440964.0151791023".

[create a15r2]
a15r2 is a area. "An area (15) in r2".
the printed name of the a15r2 is "Room 2".
Understand "Area 15 in Room 2" as a15r2.
the parent of the a15r2 is r2.
the X of the a15r2 is "547673.2968536066".
the Y of the a15r2 is "1440851.42097439".

[create a27r2]
a27r2 is a area. "An area (27) in r2".
the printed name of the a27r2 is "Room 2".
Understand "Area 27 in Room 2" as a27r2.
the parent of the a27r2 is r2.
the X of the a27r2 is "547462.7994735074".
the Y of the a27r2 is "1440820.3231596756".

[create a103r2]
a103r2 is a area. "An area (103) in r2".
the printed name of the a103r2 is "Room 2".
Understand "Area 103 in Room 2" as a103r2.
the parent of the a103r2 is r2.
the X of the a103r2 is "547714.0".
the Y of the a103r2 is "1440800.0".

[create a110r2]
a110r2 is a area. "An area (110) in r2".
the printed name of the a110r2 is "Room 2".
Understand "Area 110 in Room 2" as a110r2.
the parent of the a110r2 is r2.
the X of the a110r2 is "547432.0".
the Y of the a110r2 is "1440772.0".

[create a3r3]
a3r3 is a area. "An area (3) in r3".
the printed name of the a3r3 is "Room 3".
Understand "Area 3 in Room 3" as a3r3.
the parent of the a3r3 is r3.
the X of the a3r3 is "548225.8708448356".
the Y of the a3r3 is "1442605.316291368".

[create a40r3]
a40r3 is a area. "An area (40) in r3".
the printed name of the a40r3 is "Room 3".
Understand "Area 40 in Room 3" as a40r3.
the parent of the a40r3 is r3.
the X of the a40r3 is "548153.333333333".
the Y of the a40r3 is "1442860.9694444444".

[create a41r3]
a41r3 is a area. "An area (41) in r3".
the printed name of the a41r3 is "Room 3".
Understand "Area 41 in Room 3" as a41r3.
the parent of the a41r3 is r3.
the X of the a41r3 is "548220.5167685036".
the Y of the a41r3 is "1442773.513395759".

[create a54r3]
a54r3 is a area. "An area (54) in r3".
the printed name of the a54r3 is "Room 3".
Understand "Area 54 in Room 3" as a54r3.
the parent of the a54r3 is r3.
the X of the a54r3 is "548320.0".
the Y of the a54r3 is "1442904.0".

[create a55r3]
a55r3 is a area. "An area (55) in r3".
the printed name of the a55r3 is "Room 3".
Understand "Area 55 in Room 3" as a55r3.
the parent of the a55r3 is r3.
the X of the a55r3 is "548286.6666666665".
the Y of the a55r3 is "1442853.5364495006".

[create a107r3]
a107r3 is a area. "An area (107) in r3".
the printed name of the a107r3 is "Room 3".
Understand "Area 107 in Room 3" as a107r3.
the parent of the a107r3 is r3.
the X of the a107r3 is "548120.0".
the Y of the a107r3 is "1442904.0".

[create a6r4]
a6r4 is a area. "An area (6) in r4".
the printed name of the a6r4 is "Room 4".
Understand "Area 6 in Room 4" as a6r4.
the parent of the a6r4 is r4.
the X of the a6r4 is "548820.2390740267".
the Y of the a6r4 is "1440962.4312827692".

[create a19r4]
a19r4 is a area. "An area (19) in r4".
the printed name of the a19r4 is "Room 4".
Understand "Area 19 in Room 4" as a19r4.
the parent of the a19r4 is r4.
the X of the a19r4 is "548842.2896516462".
the Y of the a19r4 is "1440960.8833195833".

[create a20r4]
a20r4 is a area. "An area (20) in r4".
the printed name of the a20r4 is "Room 4".
Understand "Area 20 in Room 4" as a20r4.
the parent of the a20r4 is r4.
the X of the a20r4 is "548936.4539705468".
the Y of the a20r4 is "1440848.7643334023".

[create a29r4]
a29r4 is a area. "An area (29) in r4".
the printed name of the a29r4 is "Room 4".
Understand "Area 29 in Room 4" as a29r4.
the parent of the a29r4 is r4.
the X of the a29r4 is "548724.7043713275".
the Y of the a29r4 is "1440822.7067112254".

[create a100r4]
a100r4 is a area. "An area (100) in r4".
the printed name of the a100r4 is "Room 4".
Understand "Area 100 in Room 4" as a100r4.
the parent of the a100r4 is r4.
the X of the a100r4 is "548694.0".
the Y of the a100r4 is "1440772.0".

[create a106r4]
a106r4 is a area. "An area (106) in r4".
the printed name of the a106r4 is "Room 4".
Understand "Area 106 in Room 4" as a106r4.
the parent of the a106r4 is r4.
the X of the a106r4 is "548977.0".
the Y of the a106r4 is "1440800.0".

[create a4r5]
a4r5 is a area. "An area (4) in r5".
the printed name of the a4r5 is "Room 5".
Understand "Area 4 in Room 5" as a4r5.
the parent of the a4r5 is r5.
the X of the a4r5 is "549526.3900436836".
the Y of the a4r5 is "1442720.4008683984".

[create a21r5]
a21r5 is a area. "An area (21) in r5".
the printed name of the a21r5 is "Room 5".
Understand "Area 21 in Room 5" as a21r5.
the parent of the a21r5 is r5.
the X of the a21r5 is "549398.409453956".
the Y of the a21r5 is "1442730.9460312445".

[create a37r5]
a37r5 is a area. "An area (37) in r5".
the printed name of the a37r5 is "Room 5".
Understand "Area 37 in Room 5" as a37r5.
the parent of the a37r5 is r5.
the X of the a37r5 is "549017.9147631298".
the Y of the a37r5 is "1442860.132418996".

[create a38r5]
a38r5 is a area. "An area (38) in r5".
the printed name of the a38r5 is "Room 5".
Understand "Area 38 in Room 5" as a38r5.
the parent of the a38r5 is r5.
the X of the a38r5 is "549185.610276341".
the Y of the a38r5 is "1442842.8408194105".

[create a46r5]
a46r5 is a area. "An area (46) in r5".
the printed name of the a46r5 is "Room 5".
Understand "Area 46 in Room 5" as a46r5.
the parent of the a46r5 is r5.
the X of the a46r5 is "549690.0421155228".
the Y of the a46r5 is "1442827.3065045655".

[create a47r5]
a47r5 is a area. "An area (47) in r5".
the printed name of the a47r5 is "Room 5".
Understand "Area 47 in Room 5" as a47r5.
the parent of the a47r5 is r5.
the X of the a47r5 is "549924.1653756481".
the Y of the a47r5 is "1442861.2658517454".

[create a50r5]
a50r5 is a area. "An area (50) in r5".
the printed name of the a50r5 is "Room 5".
Understand "Area 50 in Room 5" as a50r5.
the parent of the a50r5 is r5.
the X of the a50r5 is "549981.0".
the Y of the a50r5 is "1442824.0".

[create a51r5]
a51r5 is a area. "An area (51) in r5".
the printed name of the a51r5 is "Room 5".
Understand "Area 51 in Room 5" as a51r5.
the parent of the a51r5 is r5.
the X of the a51r5 is "549940.6686930091".
the Y of the a51r5 is "1442842.2965214453".

[create a60r5]
a60r5 is a area. "An area (60) in r5".
the printed name of the a60r5 is "Room 5".
Understand "Area 60 in Room 5" as a60r5.
the parent of the a60r5 is r5.
the X of the a60r5 is "548971.0".
the Y of the a60r5 is "1442914.0".

[create a61r5]
a61r5 is a area. "An area (61) in r5".
the printed name of the a61r5 is "Room 5".
Understand "Area 61 in Room 5" as a61r5.
the parent of the a61r5 is r5.
the X of the a61r5 is "548999.1036507598".
the Y of the a61r5 is "1442888.4646349242".

[create a62r5]
a62r5 is a area. "An area (62) in r5".
the printed name of the a62r5 is "Room 5".
Understand "Area 62 in Room 5" as a62r5.
the parent of the a62r5 is r5.
the X of the a62r5 is "549971.0".
the Y of the a62r5 is "1442914.0".

[create a63r5]
a63r5 is a area. "An area (63) in r5".
the printed name of the a63r5 is "Room 5".
Understand "Area 63 in Room 5" as a63r5.
the parent of the a63r5 is r5.
the X of the a63r5 is "549940.1050372066".
the Y of the a63r5 is "1442887.7894485784".

[create a74r5]
a74r5 is a area. "An area (74) in r5".
the printed name of the a74r5 is "Room 5".
Understand "Area 74 in Room 5" as a74r5.
the parent of the a74r5 is r5.
the X of the a74r5 is "548962.0".
the Y of the a74r5 is "1442824.0".

[create a75r5]
a75r5 is a area. "An area (75) in r5".
the printed name of the a75r5 is "Room 5".
Understand "Area 75 in Room 5" as a75r5.
the parent of the a75r5 is r5.
the X of the a75r5 is "549002.7622699386".
the Y of the a75r5 is "1442842.6487730062".

[create a92r5]
a92r5 is a area. "An area (92) in r5".
the printed name of the a92r5 is "Room 5".
Understand "Area 92 in Room 5" as a92r5.
the parent of the a92r5 is r5.
the X of the a92r5 is "549731.0".
the Y of the a92r5 is "1442904.0".

[create a93r5]
a93r5 is a area. "An area (93) in r5".
the printed name of the a93r5 is "Room 5".
Understand "Area 93 in Room 5" as a93r5.
the parent of the a93r5 is r5.
the X of the a93r5 is "549698.005982906".
the Y of the a93r5 is "1442905.7505982905".

[create a94r5]
a94r5 is a area. "An area (94) in r5".
the printed name of the a94r5 is "Room 5".
Understand "Area 94 in Room 5" as a94r5.
the parent of the a94r5 is r5.
the X of the a94r5 is "549147.087628866".
the Y of the a94r5 is "1442904.8762886599".

[create a95r5]
a95r5 is a area. "An area (95) in r5".
the printed name of the a95r5 is "Room 5".
Understand "Area 95 in Room 5" as a95r5.
the parent of the a95r5 is r5.
the X of the a95r5 is "549180.0167107234".
the Y of the a95r5 is "1442906.1908547007".

[create a0r6]
a0r6 is a area. "An area (0) in r6".
the printed name of the a0r6 is "Room 6".
Understand "Area 0 in Room 6" as a0r6.
the parent of the a0r6 is r6.
the X of the a0r6 is "550048.7496886397".
the Y of the a0r6 is "1440962.3173736061".

[create a5r6]
a5r6 is a area. "An area (5) in r6".
the printed name of the a5r6 is "Room 6".
Understand "Area 5 in Room 6" as a5r6.
the parent of the a5r6 is r6.
the X of the a5r6 is "550034.9979687049".
the Y of the a5r6 is "1440964.011711592".

[create a16r6]
a16r6 is a area. "An area (16) in r6".
the printed name of the a16r6 is "Room 6".
Understand "Area 16 in Room 6" as a16r6.
the parent of the a16r6 is r6.
the X of the a16r6 is "550142.2968536066".
the Y of the a16r6 is "1440851.42097439".

[create a28r6]
a28r6 is a area. "An area (28) in r6".
the printed name of the a28r6 is "Room 6".
Understand "Area 28 in Room 6" as a28r6.
the parent of the a28r6 is r6.
the X of the a28r6 is "549931.8448489627".
the Y of the a28r6 is "1440819.8661640186".

[create a104r6]
a104r6 is a area. "An area (104) in r6".
the printed name of the a104r6 is "Room 6".
Understand "Area 104 in Room 6" as a104r6.
the parent of the a104r6 is r6.
the X of the a104r6 is "550183.0".
the Y of the a104r6 is "1440800.0".

[create a111r6]
a111r6 is a area. "An area (111) in r6".
the printed name of the a111r6 is "Room 6".
Understand "Area 111 in Room 6" as a111r6.
the parent of the a111r6 is r6.
the X of the a111r6 is "549901.0".
the Y of the a111r6 is "1440772.0".

[create a1r7]
a1r7 is a area. "An area (1) in r7".
the printed name of the a1r7 is "Room 7".
Understand "Area 1 in Room 7" as a1r7.
the parent of the a1r7 is r7.
the X of the a1r7 is "550694.8708448356".
the Y of the a1r7 is "1442605.316291368".

[create a42r7]
a42r7 is a area. "An area (42) in r7".
the printed name of the a42r7 is "Room 7".
Understand "Area 42 in Room 7" as a42r7.
the parent of the a42r7 is r7.
the X of the a42r7 is "550622.3333333333".
the Y of the a42r7 is "1442860.9694444442".

[create a43r7]
a43r7 is a area. "An area (43) in r7".
the printed name of the a43r7 is "Room 7".
Understand "Area 43 in Room 7" as a43r7.
the parent of the a43r7 is r7.
the X of the a43r7 is "550689.5167685037".
the Y of the a43r7 is "1442773.513395759".

[create a52r7]
a52r7 is a area. "An area (52) in r7".
the printed name of the a52r7 is "Room 7".
Understand "Area 52 in Room 7" as a52r7.
the parent of the a52r7 is r7.
the X of the a52r7 is "550789.0".
the Y of the a52r7 is "1442904.0".

[create a53r7]
a53r7 is a area. "An area (53) in r7".
the printed name of the a53r7 is "Room 7".
Understand "Area 53 in Room 7" as a53r7.
the parent of the a53r7 is r7.
the X of the a53r7 is "550755.666666667".
the Y of the a53r7 is "1442853.5364495006".

[create a108r7]
a108r7 is a area. "An area (108) in r7".
the printed name of the a108r7 is "Room 7".
Understand "Area 108 in Room 7" as a108r7.
the parent of the a108r7 is r7.
the X of the a108r7 is "550589.0".
the Y of the a108r7 is "1442904.0".

[create a7r8]
a7r8 is a area. "An area (7) in r8".
the printed name of the a7r8 is "Room 8".
Understand "Area 7 in Room 8" as a7r8.
the parent of the a7r8 is r8.
the X of the a7r8 is "551289.2390740267".
the Y of the a7r8 is "1440962.431282769".

[create a17r8]
a17r8 is a area. "An area (17) in r8".
the printed name of the a17r8 is "Room 8".
Understand "Area 17 in Room 8" as a17r8.
the parent of the a17r8 is r8.
the X of the a17r8 is "551311.289651646".
the Y of the a17r8 is "1440960.8833195835".

[create a18r8]
a18r8 is a area. "An area (18) in r8".
the printed name of the a18r8 is "Room 8".
Understand "Area 18 in Room 8" as a18r8.
the parent of the a18r8 is r8.
the X of the a18r8 is "551405.4539705467".
the Y of the a18r8 is "1440848.7643334027".

[create a30r8]
a30r8 is a area. "An area (30) in r8".
the printed name of the a30r8 is "Room 8".
Understand "Area 30 in Room 8" as a30r8.
the parent of the a30r8 is r8.
the X of the a30r8 is "551193.7043713275".
the Y of the a30r8 is "1440822.7067112252".

[create a101r8]
a101r8 is a area. "An area (101) in r8".
the printed name of the a101r8 is "Room 8".
Understand "Area 101 in Room 8" as a101r8.
the parent of the a101r8 is r8.
the X of the a101r8 is "551163.0".
the Y of the a101r8 is "1440772.0".

[create a105r8]
a105r8 is a area. "An area (105) in r8".
the printed name of the a105r8 is "Room 8".
Understand "Area 105 in Room 8" as a105r8.
the parent of the a105r8 is r8.
the X of the a105r8 is "551446.0".
the Y of the a105r8 is "1440800.0".

[create a22r9]
a22r9 is a area. "An area (22) in r9".
the printed name of the a22r9 is "Room 9".
Understand "Area 22 in Room 9" as a22r9.
the parent of the a22r9 is r9.
the X of the a22r9 is "551821.9486065779".
the Y of the a22r9 is "1442778.1064398692".

[create a31r9]
a31r9 is a area. "An area (31) in r9".
the printed name of the a31r9 is "Room 9".
Understand "Area 31 in Room 9" as a31r9.
the parent of the a31r9 is r9.
the X of the a31r9 is "551487.4921500897".
the Y of the a31r9 is "1442860.7986347189".

[create a32r9]
a32r9 is a area. "An area (32) in r9".
the printed name of the a32r9 is "Room 9".
Understand "Area 32 in Room 9" as a32r9.
the parent of the a32r9 is r9.
the X of the a32r9 is "552151.5509290386".
the Y of the a32r9 is "1442861.0062088703".

[create a66r9]
a66r9 is a area. "An area (66) in r9".
the printed name of the a66r9 is "Room 9".
Understand "Area 66 in Room 9" as a66r9.
the parent of the a66r9 is r9.
the X of the a66r9 is "551440.0".
the Y of the a66r9 is "1442914.0".

[create a67r9]
a67r9 is a area. "An area (67) in r9".
the printed name of the a67r9 is "Room 9".
Understand "Area 67 in Room 9" as a67r9.
the parent of the a67r9 is r9.
the X of the a67r9 is "551470.9153780071".
the Y of the a67r9 is "1442888.1834621993".

[create a80r9]
a80r9 is a area. "An area (80) in r9".
the printed name of the a80r9 is "Room 9".
Understand "Area 80 in Room 9" as a80r9.
the parent of the a80r9 is r9.
the X of the a80r9 is "551431.0".
the Y of the a80r9 is "1442824.0".

[create a81r9]
a81r9 is a area. "An area (81) in r9".
the printed name of the a81r9 is "Room 9".
Understand "Area 81 in Room 9" as a81r9.
the parent of the a81r9 is r9.
the X of the a81r9 is "551471.7622699386".
the Y of the a81r9 is "1442842.6487730062".

[create a82r9]
a82r9 is a area. "An area (82) in r9".
the printed name of the a82r9 is "Room 9".
Understand "Area 82 in Room 9" as a82r9.
the parent of the a82r9 is r9.
the X of the a82r9 is "552208.0".
the Y of the a82r9 is "1442824.0".

[create a83r9]
a83r9 is a area. "An area (83) in r9".
the printed name of the a83r9 is "Room 9".
Understand "Area 83 in Room 9" as a83r9.
the parent of the a83r9 is r9.
the X of the a83r9 is "552167.6625777471".
the Y of the a83r9 is "1442842.6912577748".

[create a90r9]
a90r9 is a area. "An area (90) in r9".
the printed name of the a90r9 is "Room 9".
Understand "Area 90 in Room 9" as a90r9.
the parent of the a90r9 is r9.
the X of the a90r9 is "552199.0".
the Y of the a90r9 is "1442914.0".

[create a91r9]
a91r9 is a area. "An area (91) in r9".
the printed name of the a91r9 is "Room 9".
Understand "Area 91 in Room 9" as a91r9.
the parent of the a91r9 is r9.
the X of the a91r9 is "552167.9123931624".
the Y of the a91r9 is "1442888.1662393163".

[create a10r10]
a10r10 is a area. "An area (10) in r10".
the printed name of the a10r10 is "Room 10".
Understand "Area 10 in Room 10" as a10r10.
the parent of the a10r10 is r10.
the X of the a10r10 is "545554.0365723588".
the Y of the a10r10 is "1442632.6014533543".

[create a44r10]
a44r10 is a area. "An area (44) in r10".
the printed name of the a44r10 is "Room 10".
Understand "Area 44 in Room 10" as a44r10.
the parent of the a44r10 is r10.
the X of the a44r10 is "545168.092671901".
the Y of the a44r10 is "1442859.8065726862".

[create a45r10]
a45r10 is a area. "An area (45) in r10".
the printed name of the a45r10 is "Room 10".
Understand "Area 45 in Room 10" as a45r10.
the parent of the a45r10 is r10.
the X of the a45r10 is "545503.89814202".
the Y of the a45r10 is "1442756.0890745542".

[create a64r10]
a64r10 is a area. "An area (64) in r10".
the printed name of the a64r10 is "Room 10".
Understand "Area 64 in Room 10" as a64r10.
the parent of the a64r10 is r10.
the X of the a64r10 is "545120.0".
the Y of the a64r10 is "1442914.0".

[create a65r10]
a65r10 is a area. "An area (65) in r10".
the printed name of the a65r10 is "Room 10".
Understand "Area 65 in Room 10" as a65r10.
the parent of the a65r10 is r10.
the X of the a65r10 is "545150.9153780069".
the Y of the a65r10 is "1442888.1834621993".

[create a70r10]
a70r10 is a area. "An area (70) in r10".
the printed name of the a70r10 is "Room 10".
Understand "Area 70 in Room 10" as a70r10.
the parent of the a70r10 is r10.
the X of the a70r10 is "545578.0".
the Y of the a70r10 is "1442904.0".

[create a71r10]
a71r10 is a area. "An area (71) in r10".
the printed name of the a71r10 is "Room 10".
Understand "Area 71 in Room 10" as a71r10.
the parent of the a71r10 is r10.
the X of the a71r10 is "545544.875".
the Y of the a71r10 is "1442830.000659818".

[create a78r10]
a78r10 is a area. "An area (78) in r10".
the printed name of the a78r10 is "Room 10".
Understand "Area 78 in Room 10" as a78r10.
the parent of the a78r10 is r10.
the X of the a78r10 is "545111.0".
the Y of the a78r10 is "1442824.0".

[create a79r10]
a79r10 is a area. "An area (79) in r10".
the printed name of the a79r10 is "Room 10".
Understand "Area 79 in Room 10" as a79r10.
the parent of the a79r10 is r10.
the X of the a79r10 is "545153.0599999999".
the Y of the a79r10 is "1442842.5189999999".

[create a23r11]
a23r11 is a area. "An area (23) in r11".
the printed name of the a23r11 is "Room 11".
Understand "Area 23 in Room 11" as a23r11.
the parent of the a23r11 is r11.
the X of the a23r11 is "545483.4387151967".
the Y of the a23r11 is "1440930.6903873074".

[create a24r11]
a24r11 is a area. "An area (24) in r11".
the printed name of the a24r11 is "Room 11".
Understand "Area 24 in Room 11" as a24r11.
the parent of the a24r11 is r11.
the X of the a24r11 is "545839.7586021506".
the Y of the a24r11 is "1440840.0134408602".

[create a25r11]
a25r11 is a area. "An area (25) in r11".
the printed name of the a25r11 is "Room 11".
Understand "Area 25 in Room 11" as a25r11.
the parent of the a25r11 is r11.
the X of the a25r11 is "545119.3147627445".
the Y of the a25r11 is "1440840.2898139765".

[create a68r11]
a68r11 is a area. "An area (68) in r11".
the printed name of the a68r11 is "Room 11".
Understand "Area 68 in Room 11" as a68r11.
the parent of the a68r11 is r11.
the X of the a68r11 is "545065.0".
the Y of the a68r11 is "1440880.0".

[create a69r11]
a69r11 is a area. "An area (69) in r11".
the printed name of the a69r11 is "Room 11".
Understand "Area 69 in Room 11" as a69r11.
the parent of the a69r11 is r11.
the X of the a69r11 is "545096.676963553".
the Y of the a69r11 is "1440860.7257603593".

[create a84r11]
a84r11 is a area. "An area (84) in r11".
the printed name of the a84r11 is "Room 11".
Understand "Area 84 in Room 11" as a84r11.
the parent of the a84r11 is r11.
the X of the a84r11 is "545885.0".
the Y of the a84r11 is "1440791.0".

[create a85r11]
a85r11 is a area. "An area (85) in r11".
the printed name of the a85r11 is "Room 11".
Understand "Area 85 in Room 11" as a85r11.
the parent of the a85r11 is r11.
the X of the a85r11 is "545853.7862838916".
the Y of the a85r11 is "1440816.6339712918".

[create a86r11]
a86r11 is a area. "An area (86) in r11".
the printed name of the a86r11 is "Room 11".
Understand "Area 86 in Room 11" as a86r11.
the parent of the a86r11 is r11.
the X of the a86r11 is "545074.0".
the Y of the a86r11 is "1440791.0".

[create a87r11]
a87r11 is a area. "An area (87) in r11".
the printed name of the a87r11 is "Room 11".
Understand "Area 87 in Room 11" as a87r11.
the parent of the a87r11 is r11.
the X of the a87r11 is "545105.0420577894".
the Y of the a87r11 is "1440816.6166125853".

[create a88r11]
a88r11 is a area. "An area (88) in r11".
the printed name of the a88r11 is "Room 11".
Understand "Area 88 in Room 11" as a88r11.
the parent of the a88r11 is r11.
the X of the a88r11 is "545894.0".
the Y of the a88r11 is "1440880.0".

[create a89r11]
a89r11 is a area. "An area (89) in r11".
the printed name of the a89r11 is "Room 11".
Understand "Area 89 in Room 11" as a89r11.
the parent of the a89r11 is r11.
the X of the a89r11 is "545866.849158728".
the Y of the a89r11 is "1440860.2680626004".

[create d0]
d0 is a door. "A door between a10r10 and a23r11".
d0 is south of a10r10 and north of a23r11.

[create d2]
d2 is a door. "A door between a19r4 and a21r5".
d2 is north of a19r4 and south of a21r5.

[create d4]
d4 is a door. "A door between a17r8 and a22r9".
d4 is north of a17r8 and south of a22r9.

[create d6]
d6 is a door. "A door between a0r6 and a1r7".
d6 is north of a0r6 and south of a1r7.

[create d8]
d8 is a door. "A door between a2r2 and a3r3".
d8 is north of a2r2 and south of a3r3.

[create d10]
d10 is a door. "A door between a4r5 and a5r6".
d10 is south of a4r5 and north of a5r6.

[create d12]
d12 is a door. "A door between a3r3 and a6r4".
d12 is southeast of a3r3 and northwest of a6r4.

[create d14]
d14 is a door. "A door between a1r7 and a7r8".
d14 is southeast of a1r7 and northwest of a7r8.

[create d16]
d16 is a door. "A door between a8r1 and a9r2".
d16 is south of a8r1 and north of a9r2.

[create d18]
d18 is a door. "A door between a11r0 and a10r10".
d18 is northwest of a11r0 and southeast of a10r10.

[create d20]
d20 is a door. "A door between a11r0 and a12r1".
d20 is north of a11r0 and south of a12r1.

northeast of a13r0 is southwest of a14r0.

southeast of a2r2 is northwest of a15r2.

southeast of a0r6 is northwest of a16r6.

southeast of a17r8 is northwest of a18r8.

southeast of a19r4 is northwest of a20r4.

south of a11r0 is north of a14r0.

east of a23r11 is west of a24r11.

west of a23r11 is east of a25r11.

southeast of a14r0 is northwest of a26r0.

southwest of a9r2 is northeast of a27r2.

southwest of a5r6 is northeast of a28r6.

southwest of a6r4 is northeast of a29r4.

southwest of a7r8 is northeast of a30r8.

west of a22r9 is east of a31r9.

east of a22r9 is west of a32r9.

east of a33r1 is west of a34r1.

east of a35r1 is west of a36r1.

east of a37r5 is west of a38r5.

southwest of a35r1 is northeast of a39r1.

southeast of a34r1 is northwest of a39r1.

west of a4r5 is east of a21r5.

northwest of a21r5 is southeast of a38r5.

southeast of a40r3 is northwest of a41r3.

southeast of a42r7 is northwest of a43r7.

east of a44r10 is west of a45r10.

east of a46r5 is west of a47r5.

north of a12r1 is south of a48r1.

north of a8r1 is south of a49r1.

southeast of a48r1 is northwest of a49r1.

south of a39r1 is north of a48r1.

north of a3r3 is south of a41r3.

north of a1r7 is south of a43r7.

north of a10r10 is south of a45r10.

northeast of a4r5 is southwest of a46r5.

northwest of a50r5 is southeast of a51r5.

southwest of a52r7 is northeast of a53r7.

southwest of a54r3 is northeast of a55r3.

southeast of a56r1 is northwest of a57r1.

southwest of a58r1 is northeast of a59r1.

southeast of a60r5 is northwest of a61r5.

southwest of a62r5 is northeast of a63r5.

southeast of a64r10 is northwest of a65r10.

southeast of a66r9 is northwest of a67r9.

southeast of a68r11 is northwest of a69r11.

southwest of a70r10 is northeast of a71r10.

northwest of a72r1 is southeast of a73r1.

northeast of a74r5 is southwest of a75r5.

northeast of a76r1 is southwest of a77r1.

northeast of a78r10 is southwest of a79r10.

northeast of a80r9 is southwest of a81r9.

northwest of a82r9 is southeast of a83r9.

northwest of a84r11 is southeast of a85r11.

northeast of a86r11 is southwest of a87r11.

southwest of a88r11 is northeast of a89r11.

southwest of a90r9 is northeast of a91r9.

west of a92r5 is east of a93r5.

east of a94r5 is west of a95r5.

west of a96r1 is east of a97r1.

east of a98r1 is west of a99r1.

southwest of a29r4 is northeast of a100r4.

southwest of a30r8 is northeast of a101r8.

south of a13r0 is north of a102r0.

southeast of a15r2 is northwest of a103r2.

southeast of a16r6 is northwest of a104r6.

southeast of a18r8 is northwest of a105r8.

southeast of a20r4 is northwest of a106r4.

northwest of a40r3 is southeast of a107r3.

northwest of a42r7 is southeast of a108r7.

southeast of a26r0 is northwest of a109r0.

southwest of a27r2 is northeast of a110r2.

southwest of a28r6 is northeast of a111r6.

northeast of a41r3 is southwest of a55r3.

northwest of a33r1 is southeast of a57r1.

northeast of a36r1 is southwest of a59r1.

northwest of a37r5 is southeast of a61r5.

northeast of a47r5 is southwest of a63r5.

northwest of a44r10 is southeast of a65r10.

northwest of a31r9 is southeast of a67r9.

northeast of a32r9 is southwest of a91r9.

north of a46r5 is south of a93r5.

southeast of a36r1 is northwest of a73r1.

southwest of a37r5 is northeast of a75r5.

southwest of a33r1 is northeast of a77r1.

southwest of a44r10 is northeast of a79r10.

southwest of a31r9 is northeast of a81r9.

southeast of a32r9 is northwest of a83r9.

southeast of a47r5 is northwest of a51r5.

northeast of a43r7 is southwest of a53r7.

southwest of a25r11 is northeast of a87r11.

northeast of a24r11 is southwest of a89r11.

northwest of a25r11 is southeast of a69r11.

east of a7r8 is west of a17r8.

east of a6r4 is west of a19r4.

west of a0r6 is east of a5r6.

west of a2r2 is east of a9r2.

northeast of a45r10 is southwest of a71r10.

southeast of a24r11 is northwest of a85r11.

north of a38r5 is south of a95r5.

north of a35r1 is south of a97r1.

north of a34r1 is south of a99r1.

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
the player is in a82r9.

the orientation of the player is 6.
