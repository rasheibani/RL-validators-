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

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "545370.4497203713".
the Y of the a6r0 is "2183894.2861943264".

[create a31r0]
a31r0 is a area. "An area (31) in r0".
the printed name of the a31r0 is "Room 0".
Understand "Area 31 in Room 0" as a31r0.
the parent of the a31r0 is r0.
the X of the a31r0 is "545261.3750000002".
the Y of the a31r0 is "2184001.379310345".

[create a34r0]
a34r0 is a area. "An area (34) in r0".
the printed name of the a34r0 is "Room 0".
Understand "Area 34 in Room 0" as a34r0.
the parent of the a34r0 is r0.
the X of the a34r0 is "545479.6250000005".
the Y of the a34r0 is "2184001.8461538465".

[create a55r0]
a55r0 is a area. "An area (55) in r0".
the printed name of the a55r0 is "Room 0".
Understand "Area 55 in Room 0" as a55r0.
the parent of the a55r0 is r0.
the X of the a55r0 is "545225.0".
the Y of the a55r0 is "2184040.0".

[create a58r0]
a58r0 is a area. "An area (58) in r0".
the printed name of the a58r0 is "Room 0".
Understand "Area 58 in Room 0" as a58r0.
the parent of the a58r0 is r0.
the X of the a58r0 is "545516.0".
the Y of the a58r0 is "2184040.0".

[create a0r1]
a0r1 is a area. "An area (0) in r1".
the printed name of the a0r1 is "Room 1".
Understand "Area 0 in Room 1" as a0r1.
the parent of the a0r1 is r1.
the X of the a0r1 is "545263.1666666667".
the Y of the a0r1 is "2181838.620689655".

[create a1r1]
a1r1 is a area. "An area (1) in r1".
the printed name of the a1r1 is "Room 1".
Understand "Area 1 in Room 1" as a1r1.
the parent of the a1r1 is r1.
the X of the a1r1 is "545385.9734857862".
the Y of the a1r1 is "2181960.8865061617".

[create a2r1]
a2r1 is a area. "An area (2) in r1".
the printed name of the a2r1 is "Room 1".
Understand "Area 2 in Room 1" as a2r1.
the parent of the a2r1 is r1.
the X of the a2r1 is "546468.8494153669".
the Y of the a2r1 is "2181928.188284103".

[create a3r1]
a3r1 is a area. "An area (3) in r1".
the printed name of the a3r1 is "Room 1".
Understand "Area 3 in Room 1" as a3r1.
the parent of the a3r1 is r1.
the X of the a3r1 is "546560.8333333333".
the Y of the a3r1 is "2181832.0".

[create a21r1]
a21r1 is a area. "An area (21) in r1".
the printed name of the a21r1 is "Room 1".
Understand "Area 21 in Room 1" as a21r1.
the parent of the a21r1 is r1.
the X of the a21r1 is "546587.7949339207".
the Y of the a21r1 is "2182024.0".

[create a45r1]
a45r1 is a area. "An area (45) in r1".
the printed name of the a45r1 is "Room 1".
Understand "Area 45 in Room 1" as a45r1.
the parent of the a45r1 is r1.
the X of the a45r1 is "545225.0".
the Y of the a45r1 is "2181800.0".

[create a53r1]
a53r1 is a area. "An area (53) in r1".
the printed name of the a53r1 is "Room 1".
Understand "Area 53 in Room 1" as a53r1.
the parent of the a53r1 is r1.
the X of the a53r1 is "546599.0".
the Y of the a53r1 is "2181800.0".

[create a11r2]
a11r2 is a area. "An area (11) in r2".
the printed name of the a11r2 is "Room 2".
Understand "Area 11 in Room 2" as a11r2.
the parent of the a11r2 is r2.
the X of the a11r2 is "546282.1371343825".
the Y of the a11r2 is "2183898.0705300597".

[create a35r2]
a35r2 is a area. "An area (35) in r2".
the printed name of the a35r2 is "Room 2".
Understand "Area 35 in Room 2" as a35r2.
the parent of the a35r2 is r2.
the X of the a35r2 is "546357.7499999999".
the Y of the a35r2 is "2183992.81659307".

[create a43r2]
a43r2 is a area. "An area (43) in r2".
the printed name of the a43r2 is "Room 2".
Understand "Area 43 in Room 2" as a43r2.
the parent of the a43r2 is r2.
the X of the a43r2 is "546134.2500000002".
the Y of the a43r2 is "2184009.8327083336".

[create a59r2]
a59r2 is a area. "An area (59) in r2".
the printed name of the a59r2 is "Room 2".
Understand "Area 59 in Room 2" as a59r2.
the parent of the a59r2 is r2.
the X of the a59r2 is "546395.0".
the Y of the a59r2 is "2184040.0".

[create a60r2]
a60r2 is a area. "An area (60) in r2".
the printed name of the a60r2 is "Room 2".
Understand "Area 60 in Room 2" as a60r2.
the parent of the a60r2 is r2.
the X of the a60r2 is "546097.0".
the Y of the a60r2 is "2184040.0".

[create a7r3]
a7r3 is a area. "An area (7) in r3".
the printed name of the a7r3 is "Room 3".
Understand "Area 7 in Room 3" as a7r3.
the parent of the a7r3 is r3.
the X of the a7r3 is "546859.9412801261".
the Y of the a7r3 is "2181954.6105585736".

[create a12r3]
a12r3 is a area. "An area (12) in r3".
the printed name of the a12r3 is "Room 3".
Understand "Area 12 in Room 3" as a12r3.
the parent of the a12r3 is r3.
the X of the a12r3 is "546713.2943062792".
the Y of the a12r3 is "2182087.701323167".

[create a22r3]
a22r3 is a area. "An area (22) in r3".
the printed name of the a22r3 is "Room 3".
Understand "Area 22 in Room 3" as a22r3.
the parent of the a22r3 is r3.
the X of the a22r3 is "546939.875".
the Y of the a22r3 is "2181850.099474627".

[create a46r3]
a46r3 is a area. "An area (46) in r3".
the printed name of the a46r3 is "Room 3".
Understand "Area 46 in Room 3" as a46r3.
the parent of the a46r3 is r3.
the X of the a46r3 is "546656.905102717".
the Y of the a46r3 is "2182073.02502599".

[create a47r3]
a47r3 is a area. "An area (47) in r3".
the printed name of the a47r3 is "Room 3".
Understand "Area 47 in Room 3" as a47r3.
the parent of the a47r3 is r3.
the X of the a47r3 is "546977.0".
the Y of the a47r3 is "2181800.0".

[create a83r3]
a83r3 is a area. "An area (83) in r3".
the printed name of the a83r3 is "Room 3".
Understand "Area 83 in Room 3" as a83r3.
the parent of the a83r3 is r3.
the X of the a83r3 is "546743.3032067444".
the Y of the a83r3 is "2182015.787136696".

[create a84r3]
a84r3 is a area. "An area (84) in r3".
the printed name of the a84r3 is "Room 3".
Understand "Area 84 in Room 3" as a84r3.
the parent of the a84r3 is r3.
the X of the a84r3 is "546797.6462401494".
the Y of the a84r3 is "2181953.4981471454".

[create a8r4]
a8r4 is a area. "An area (8) in r4".
the printed name of the a8r4 is "Room 4".
Understand "Area 8 in Room 4" as a8r4.
the parent of the a8r4 is r4.
the X of the a8r4 is "547446.5565103935".
the Y of the a8r4 is "2183902.537885611".

[create a25r4]
a25r4 is a area. "An area (25) in r4".
the printed name of the a25r4 is "Room 4".
Understand "Area 25 in Room 4" as a25r4.
the parent of the a25r4 is r4.
the X of the a25r4 is "547582.830598208".
the Y of the a25r4 is "2183898.894703835".

[create a32r4]
a32r4 is a area. "An area (32) in r4".
the printed name of the a32r4 is "Room 4".
Understand "Area 32 in Room 4" as a32r4.
the parent of the a32r4 is r4.
the X of the a32r4 is "547375.0999999999".
the Y of the a32r4 is "2183989.523978247".

[create a33r4]
a33r4 is a area. "An area (33) in r4".
the printed name of the a33r4 is "Room 4".
Understand "Area 33 in Room 4" as a33r4.
the parent of the a33r4 is r4.
the X of the a33r4 is "547655.8999999999".
the Y of the a33r4 is "2183989.466754617".

[create a56r4]
a56r4 is a area. "An area (56) in r4".
the printed name of the a56r4 is "Room 4".
Understand "Area 56 in Room 4" as a56r4.
the parent of the a56r4 is r4.
the X of the a56r4 is "547340.0".
the Y of the a56r4 is "2184040.0".

[create a57r4]
a57r4 is a area. "An area (57) in r4".
the printed name of the a57r4 is "Room 4".
Understand "Area 57 in Room 4" as a57r4.
the parent of the a57r4 is r4.
the X of the a57r4 is "547691.0".
the Y of the a57r4 is "2184040.0".

[create a19r5]
a19r5 is a area. "An area (19) in r5".
the printed name of the a19r5 is "Room 5".
Understand "Area 19 in Room 5" as a19r5.
the parent of the a19r5 is r5.
the X of the a19r5 is "548204.9255680316".
the Y of the a19r5 is "2181954.3366956883".

[create a20r5]
a20r5 is a area. "An area (20) in r5".
the printed name of the a20r5 is "Room 5".
Understand "Area 20 in Room 5" as a20r5.
the parent of the a20r5 is r5.
the X of the a20r5 is "548286.0".
the Y of the a20r5 is "2181849.261434642".

[create a23r5]
a23r5 is a area. "An area (23) in r5".
the printed name of the a23r5 is "Room 5".
Understand "Area 23 in Room 5" as a23r5.
the parent of the a23r5 is r5.
the X of the a23r5 is "548076.0".
the Y of the a23r5 is "2181849.422623434".

[create a24r5]
a24r5 is a area. "An area (24) in r5".
the printed name of the a24r5 is "Room 5".
Understand "Area 24 in Room 5" as a24r5.
the parent of the a24r5 is r5.
the X of the a24r5 is "548156.2576810804".
the Y of the a24r5 is "2181953.7046452085".

[create a44r5]
a44r5 is a area. "An area (44) in r5".
the printed name of the a44r5 is "Room 5".
Understand "Area 44 in Room 5" as a44r5.
the parent of the a44r5 is r5.
the X of the a44r5 is "548321.0".
the Y of the a44r5 is "2181800.0".

[create a48r5]
a48r5 is a area. "An area (48) in r5".
the printed name of the a48r5 is "Room 5".
Understand "Area 48 in Room 5" as a48r5.
the parent of the a48r5 is r5.
the X of the a48r5 is "548041.0".
the Y of the a48r5 is "2181800.0".

[create a9r6]
a9r6 is a area. "An area (9) in r6".
the printed name of the a9r6 is "Room 6".
Understand "Area 9 in Room 6" as a9r6.
the parent of the a9r6 is r6.
the X of the a9r6 is "548830.460768834".
the Y of the a9r6 is "2183036.656716418".

[create a16r6]
a16r6 is a area. "An area (16) in r6".
the printed name of the a16r6 is "Room 6".
Understand "Area 16 in Room 6" as a16r6.
the parent of the a16r6 is r6.
the X of the a16r6 is "548547.5386871184".
the Y of the a16r6 is "2183133.401181974".

[create a17r6]
a17r6 is a area. "An area (17) in r6".
the printed name of the a17r6 is "Room 6".
Understand "Area 17 in Room 6" as a17r6.
the parent of the a17r6 is r6.
the X of the a17r6 is "548528.4341612107".
the Y of the a17r6 is "2183058.864932833".

[create a26r6]
a26r6 is a area. "An area (26) in r6".
the printed name of the a26r6 is "Room 6".
Understand "Area 26 in Room 6" as a26r6.
the parent of the a26r6 is r6.
the X of the a26r6 is "549051.1539155536".
the Y of the a26r6 is "2183675.8094368116".

[create a27r6]
a27r6 is a area. "An area (27) in r6".
the printed name of the a27r6 is "Room 6".
Understand "Area 27 in Room 6" as a27r6.
the parent of the a27r6 is r6.
the X of the a27r6 is "550004.5314170668".
the Y of the a27r6 is "2183903.1892990773".

[create a28r6]
a28r6 is a area. "An area (28) in r6".
the printed name of the a28r6 is "Room 6".
Understand "Area 28 in Room 6" as a28r6.
the parent of the a28r6 is r6.
the X of the a28r6 is "550370.2356620994".
the Y of the a28r6 is "2183614.331910424".

[create a29r6]
a29r6 is a area. "An area (29) in r6".
the printed name of the a29r6 is "Room 6".
Understand "Area 29 in Room 6" as a29r6.
the parent of the a29r6 is r6.
the X of the a29r6 is "548742.4609628965".
the Y of the a29r6 is "2183901.4317349754".

[create a30r6]
a30r6 is a area. "An area (30) in r6".
the printed name of the a30r6 is "Room 6".
Understand "Area 30 in Room 6" as a30r6.
the parent of the a30r6 is r6.
the X of the a30r6 is "548887.4999999999".
the Y of the a30r6 is "2184012.951645433".

[create a36r6]
a36r6 is a area. "An area (36) in r6".
the printed name of the a36r6 is "Room 6".
Understand "Area 36 in Room 6" as a36r6.
the parent of the a36r6 is r6.
the X of the a36r6 is "548632.2774717972".
the Y of the a36r6 is "2183302.613795182".

[create a37r6]
a37r6 is a area. "An area (37) in r6".
the printed name of the a37r6 is "Room 6".
Understand "Area 37 in Room 6" as a37r6.
the parent of the a37r6 is r6.
the X of the a37r6 is "548824.4738833166".
the Y of the a37r6 is "2183250.8752076174".

[create a38r6]
a38r6 is a area. "An area (38) in r6".
the printed name of the a38r6 is "Room 6".
Understand "Area 38 in Room 6" as a38r6.
the parent of the a38r6 is r6.
the X of the a38r6 is "548639.7566097162".
the Y of the a38r6 is "2183489.65889394".

[create a39r6]
a39r6 is a area. "An area (39) in r6".
the printed name of the a39r6 is "Room 6".
Understand "Area 39 in Room 6" as a39r6.
the parent of the a39r6 is r6.
the X of the a39r6 is "550400.1924349968".
the Y of the a39r6 is "2183512.199723975".

[create a40r6]
a40r6 is a area. "An area (40) in r6".
the printed name of the a40r6 is "Room 6".
Understand "Area 40 in Room 6" as a40r6.
the parent of the a40r6 is r6.
the X of the a40r6 is "550543.6090407216".
the Y of the a40r6 is "2183431.103810641".

[create a41r6]
a41r6 is a area. "An area (41) in r6".
the printed name of the a41r6 is "Room 6".
Understand "Area 41 in Room 6" as a41r6.
the parent of the a41r6 is r6.
the X of the a41r6 is "550329.1472802768".
the Y of the a41r6 is "2183377.4773217994".

[create a42r6]
a42r6 is a area. "An area (42) in r6".
the printed name of the a42r6 is "Room 6".
Understand "Area 42 in Room 6" as a42r6.
the parent of the a42r6 is r6.
the X of the a42r6 is "548668.4999999994".
the Y of the a42r6 is "2183991.129062261".

[create a51r6]
a51r6 is a area. "An area (51) in r6".
the printed name of the a51r6 is "Room 6".
Understand "Area 51 in Room 6" as a51r6.
the parent of the a51r6 is r6.
the X of the a51r6 is "550588.0".
the Y of the a51r6 is "2183403.0".

[create a52r6]
a52r6 is a area. "An area (52) in r6".
the printed name of the a52r6 is "Room 6".
Understand "Area 52 in Room 6" as a52r6.
the parent of the a52r6 is r6.
the X of the a52r6 is "550301.0".
the Y of the a52r6 is "2183336.0".

[create a54r6]
a54r6 is a area. "An area (54) in r6".
the printed name of the a54r6 is "Room 6".
Understand "Area 54 in Room 6" as a54r6.
the parent of the a54r6 is r6.
the X of the a54r6 is "548924.0".
the Y of the a54r6 is "2184040.0".

[create a61r6]
a61r6 is a area. "An area (61) in r6".
the printed name of the a61r6 is "Room 6".
Understand "Area 61 in Room 6" as a61r6.
the parent of the a61r6 is r6.
the X of the a61r6 is "548632.0".
the Y of the a61r6 is "2184040.0".

[create a62r6]
a62r6 is a area. "An area (62) in r6".
the printed name of the a62r6 is "Room 6".
Understand "Area 62 in Room 6" as a62r6.
the parent of the a62r6 is r6.
the X of the a62r6 is "548944.0".
the Y of the a62r6 is "2183778.0".

[create a63r6]
a63r6 is a area. "An area (63) in r6".
the printed name of the a63r6 is "Room 6".
Understand "Area 63 in Room 6" as a63r6.
the parent of the a63r6 is r6.
the X of the a63r6 is "549034.9423265156".
the Y of the a63r6 is "2183692.578733788".

[create a64r6]
a64r6 is a area. "An area (64) in r6".
the printed name of the a64r6 is "Room 6".
Understand "Area 64 in Room 6" as a64r6.
the parent of the a64r6 is r6.
the X of the a64r6 is "550499.0".
the Y of the a64r6 is "2183706.0".

[create a65r6]
a65r6 is a area. "An area (65) in r6".
the printed name of the a65r6 is "Room 6".
Understand "Area 65 in Room 6" as a65r6.
the parent of the a65r6 is r6.
the X of the a65r6 is "550404.1699148411".
the Y of the a65r6 is "2183637.002879277".

[create a66r6]
a66r6 is a area. "An area (66) in r6".
the printed name of the a66r6 is "Room 6".
Understand "Area 66 in Room 6" as a66r6.
the parent of the a66r6 is r6.
the X of the a66r6 is "550049.0".
the Y of the a66r6 is "2184040.0".

[create a67r6]
a67r6 is a area. "An area (67) in r6".
the printed name of the a67r6 is "Room 6".
Understand "Area 67 in Room 6" as a67r6.
the parent of the a67r6 is r6.
the X of the a67r6 is "550016.0".
the Y of the a67r6 is "2183936.7709580837".

[create a76r6]
a76r6 is a area. "An area (76) in r6".
the printed name of the a76r6 is "Room 6".
Understand "Area 76 in Room 6" as a76r6.
the parent of the a76r6 is r6.
the X of the a76r6 is "548681.0".
the Y of the a76r6 is "2182936.0".

[create a77r6]
a77r6 is a area. "An area (77) in r6".
the printed name of the a77r6 is "Room 6".
Understand "Area 77 in Room 6" as a77r6.
the parent of the a77r6 is r6.
the X of the a77r6 is "548650.1468610526".
the Y of the a77r6 is "2183026.100308724".

[create a78r6]
a78r6 is a area. "An area (78) in r6".
the printed name of the a78r6 is "Room 6".
Understand "Area 78 in Room 6" as a78r6.
the parent of the a78r6 is r6.
the X of the a78r6 is "548681.7705672851".
the Y of the a78r6 is "2183036.656716418".

[create a79r6]
a79r6 is a area. "An area (79) in r6".
the printed name of the a79r6 is "Room 6".
Understand "Area 79 in Room 6" as a79r6.
the parent of the a79r6 is r6.
the X of the a79r6 is "548757.3986515565".
the Y of the a79r6 is "2183421.1160522164".

[create a80r6]
a80r6 is a area. "An area (80) in r6".
the printed name of the a80r6 is "Room 6".
Understand "Area 80 in Room 6" as a80r6.
the parent of the a80r6 is r6.
the X of the a80r6 is "548761.3285391824".
the Y of the a80r6 is "2183456.9321647123".

[create a81r6]
a81r6 is a area. "An area (81) in r6".
the printed name of the a81r6 is "Room 6".
Understand "Area 81 in Room 6" as a81r6.
the parent of the a81r6 is r6.
the X of the a81r6 is "548681.0".
the Y of the a81r6 is "2183137.313432836".

[create a82r6]
a82r6 is a area. "An area (82) in r6".
the printed name of the a82r6 is "Room 6".
Understand "Area 82 in Room 6" as a82r6.
the parent of the a82r6 is r6.
the X of the a82r6 is "548657.6441149698".
the Y of the a82r6 is "2183103.7611940303".

[create a4r7]
a4r7 is a area. "An area (4) in r7".
the printed name of the a4r7 is "Room 7".
Understand "Area 4 in Room 7" as a4r7.
the parent of the a4r7 is r7.
the X of the a4r7 is "549399.6022888334".
the Y of the a4r7 is "2181942.1318872175".

[create a5r7]
a5r7 is a area. "An area (5) in r7".
the printed name of the a5r7 is "Room 7".
Understand "Area 5 in Room 7" as a5r7.
the parent of the a5r7 is r7.
the X of the a5r7 is "550012.9665046118".
the Y of the a5r7 is "2181938.78788223".

[create a10r7]
a10r7 is a area. "An area (10) in r7".
the printed name of the a10r7 is "Room 7".
Understand "Area 10 in Room 7" as a10r7.
the parent of the a10r7 is r7.
the X of the a10r7 is "549025.3114406404".
the Y of the a10r7 is "2182178.9584036255".

[create a13r7]
a13r7 is a area. "An area (13) in r7".
the printed name of the a13r7 is "Room 7".
Understand "Area 13 in Room 7" as a13r7.
the parent of the a13r7 is r7.
the X of the a13r7 is "550442.6635692205".
the Y of the a13r7 is "2182387.517056374".

[create a14r7]
a14r7 is a area. "An area (14) in r7".
the printed name of the a14r7 is "Room 7".
Understand "Area 14 in Room 7" as a14r7.
the parent of the a14r7 is r7.
the X of the a14r7 is "550580.2335566808".
the Y of the a14r7 is "2182470.6741780005".

[create a15r7]
a15r7 is a area. "An area (15) in r7".
the printed name of the a15r7 is "Room 7".
Understand "Area 15 in Room 7" as a15r7.
the parent of the a15r7 is r7.
the X of the a15r7 is "550404.6555996799".
the Y of the a15r7 is "2182242.855297461".

[create a18r7]
a18r7 is a area. "An area (18) in r7".
the printed name of the a18r7 is "Room 7".
Understand "Area 18 in Room 7" as a18r7.
the parent of the a18r7 is r7.
the X of the a18r7 is "550362.5623291695".
the Y of the a18r7 is "2182527.7313395664".

[create a49r7]
a49r7 is a area. "An area (49) in r7".
the printed name of the a49r7 is "Room 7".
Understand "Area 49 in Room 7" as a49r7.
the parent of the a49r7 is r7.
the X of the a49r7 is "550334.0".
the Y of the a49r7 is "2182568.0".

[create a50r7]
a50r7 is a area. "An area (50) in r7".
the printed name of the a50r7 is "Room 7".
Understand "Area 50 in Room 7" as a50r7.
the parent of the a50r7 is r7.
the X of the a50r7 is "550625.0".
the Y of the a50r7 is "2182495.0".

[create a68r7]
a68r7 is a area. "An area (68) in r7".
the printed name of the a68r7 is "Room 7".
Understand "Area 68 in Room 7" as a68r7.
the parent of the a68r7 is r7.
the X of the a68r7 is "549359.0".
the Y of the a68r7 is "2181800.0".

[create a69r7]
a69r7 is a area. "An area (69) in r7".
the printed name of the a69r7 is "Room 7".
Understand "Area 69 in Room 7" as a69r7.
the parent of the a69r7 is r7.
the X of the a69r7 is "549394.6".
the Y of the a69r7 is "2181925.5734335845".

[create a70r7]
a70r7 is a area. "An area (70) in r7".
the printed name of the a70r7 is "Room 7".
Understand "Area 70 in Room 7" as a70r7.
the parent of the a70r7 is r7.
the X of the a70r7 is "550058.0".
the Y of the a70r7 is "2181800.0".

[create a71r7]
a71r7 is a area. "An area (71) in r7".
the printed name of the a71r7 is "Room 7".
Understand "Area 71 in Room 7" as a71r7.
the parent of the a71r7 is r7.
the X of the a71r7 is "550023.7".
the Y of the a71r7 is "2181907.8898876403".

[create a72r7]
a72r7 is a area. "An area (72) in r7".
the printed name of the a72r7 is "Room 7".
Understand "Area 72 in Room 7" as a72r7.
the parent of the a72r7 is r7.
the X of the a72r7 is "550534.0".
the Y of the a72r7 is "2182156.0".

[create a73r7]
a73r7 is a area. "An area (73) in r7".
the printed name of the a73r7 is "Room 7".
Understand "Area 73 in Room 7" as a73r7.
the parent of the a73r7 is r7.
the X of the a73r7 is "550441.9477917004".
the Y of the a73r7 is "2182217.0529526705".

[create a74r7]
a74r7 is a area. "An area (74) in r7".
the printed name of the a74r7 is "Room 7".
Understand "Area 74 in Room 7" as a74r7.
the parent of the a74r7 is r7.
the X of the a74r7 is "548914.0".
the Y of the a74r7 is "2182085.0".

[create a75r7]
a75r7 is a area. "An area (75) in r7".
the printed name of the a75r7 is "Room 7".
Understand "Area 75 in Room 7" as a75r7.
the parent of the a75r7 is r7.
the X of the a75r7 is "549011.3466845362".
the Y of the a75r7 is "2182167.0099861315".

[create d0]
d0 is a door. "A door between a25r4 and a24r5".
d0 is south of a25r4 and north of a24r5.

[create d2]
d2 is a door. "A door between a19r5 and a17r6".
d2 is north of a19r5 and south of a17r6.

[create d4]
d4 is a door. "A door between a7r3 and a8r4".
d4 is north of a7r3 and south of a8r4.

[create d6]
d6 is a door. "A door between a6r0 and a1r1".
d6 is south of a6r0 and north of a1r1.

[create d8]
d8 is a door. "A door between a11r2 and a12r3".
d8 is south of a11r2 and north of a12r3.

[create d10]
d10 is a door. "A door between a9r6 and a10r7".
d10 is south of a9r6 and north of a10r7.

northeast of a0r1 is southwest of a1r1.

southeast of a2r1 is northwest of a3r1.

east of a1r1 is west of a2r1.

east of a4r7 is west of a5r7.

northeast of a13r7 is southwest of a14r7.

south of a13r7 is north of a15r7.

south of a16r6 is north of a17r6.

northwest of a13r7 is southeast of a18r7.

southeast of a19r5 is northwest of a20r5.

northeast of a2r1 is southwest of a21r1.

southeast of a7r3 is northwest of a22r3.

northeast of a23r5 is southwest of a24r5.

northeast of a5r7 is southwest of a15r7.

northwest of a4r7 is southeast of a10r7.

east of a26r6 is west of a27r6.

southeast of a27r6 is northwest of a28r6.

northeast of a29r6 is southwest of a30r6.

northwest of a6r0 is southeast of a31r0.

northwest of a8r4 is southeast of a32r4.

northeast of a25r4 is southwest of a33r4.

northeast of a6r0 is southwest of a34r0.

northeast of a11r2 is southwest of a35r2.

northeast of a16r6 is southwest of a36r6.

north of a9r6 is south of a37r6.

north of a36r6 is south of a38r6.

southwest of a26r6 is northeast of a37r6.

southeast of a39r6 is northwest of a40r6.

southwest of a39r6 is northeast of a41r6.

south of a28r6 is north of a39r6.

south of a29r6 is north of a38r6.

northwest of a29r6 is southeast of a42r6.

east of a8r4 is west of a25r4.

northwest of a11r2 is southeast of a43r2.

southeast of a20r5 is northwest of a44r5.

southwest of a0r1 is northeast of a45r1.

southeast of a22r3 is northwest of a47r3.

southwest of a23r5 is northeast of a48r5.

northwest of a18r7 is southeast of a49r7.

northeast of a14r7 is southwest of a50r7.

southeast of a40r6 is northwest of a51r6.

southwest of a41r6 is northeast of a52r6.

southeast of a3r1 is northwest of a53r1.

northeast of a30r6 is southwest of a54r6.

northwest of a31r0 is southeast of a55r0.

northwest of a32r4 is southeast of a56r4.

northeast of a33r4 is southwest of a57r4.

northeast of a34r0 is southwest of a58r0.

northeast of a35r2 is southwest of a59r2.

northwest of a43r2 is southeast of a60r2.

northwest of a42r6 is southeast of a61r6.

southeast of a62r6 is northwest of a63r6.

southwest of a64r6 is northeast of a65r6.

south of a66r6 is north of a67r6.

north of a68r7 is south of a69r7.

north of a70r7 is south of a71r7.

northwest of a72r7 is southeast of a73r7.

northeast of a74r7 is southwest of a75r7.

north of a76r6 is south of a77r6.

north of a76r6 is south of a78r6.

north of a79r6 is south of a80r6.

southwest of a81r6 is northeast of a82r6.

west of a19r5 is east of a24r5.

south of a4r7 is north of a69r7.

southeast of a83r3 is northwest of a84r3.

west of a7r3 is east of a84r3.

southeast of a12r3 is northwest of a83r3.

west of a12r3 is east of a46r3.

southeast of a15r7 is northwest of a73r7.

southwest of a10r7 is northeast of a75r7.

south of a5r7 is north of a71r7.

east of a17r6 is west of a77r6.

west of a9r6 is east of a78r6.

east of a36r6 is west of a37r6.

east of a16r6 is west of a82r6.

northeast of a28r6 is southwest of a65r6.

east of a38r6 is west of a80r6.

north of a27r6 is south of a67r6.

northwest of a26r6 is southeast of a63r6.

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
the player is in a79r6.

the orientation of the player is 6.
