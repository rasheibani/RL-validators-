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

[create a10r0]
a10r0 is a area. "An area (10) in r0".
the printed name of the a10r0 is "Room 0".
Understand "Area 10 in Room 0" as a10r0.
the parent of the a10r0 is r0.
the X of the a10r0 is "546001.0".
the Y of the a10r0 is "1488514.0".

[create a11r0]
a11r0 is a area. "An area (11) in r0".
the printed name of the a11r0 is "Room 0".
Understand "Area 11 in Room 0" as a11r0.
the parent of the a11r0 is r0.
the X of the a11r0 is "546032.8264604812".
the Y of the a11r0 is "1488505.767353952".

[create a12r0]
a12r0 is a area. "An area (12) in r0".
the printed name of the a12r0 is "Room 0".
Understand "Area 12 in Room 0" as a12r0.
the parent of the a12r0 is r0.
the X of the a12r0 is "546693.0".
the Y of the a12r0 is "1488179.0".

[create a13r0]
a13r0 is a area. "An area (13) in r0".
the printed name of the a13r0 is "Room 0".
Understand "Area 13 in Room 0" as a13r0.
the parent of the a13r0 is r0.
the X of the a13r0 is "546625.3004502935".
the Y of the a13r0 is "1488153.3325991188".

[create a14r0]
a14r0 is a area. "An area (14) in r0".
the printed name of the a14r0 is "Room 0".
Understand "Area 14 in Room 0" as a14r0.
the parent of the a14r0 is r0.
the X of the a14r0 is "545111.0".
the Y of the a14r0 is "1488424.0".

[create a15r0]
a15r0 is a area. "An area (15) in r0".
the printed name of the a15r0 is "Room 0".
Understand "Area 15 in Room 0" as a15r0.
the parent of the a15r0 is r0.
the X of the a15r0 is "545153.06".
the Y of the a15r0 is "1488442.5189999999".

[create a16r0]
a16r0 is a area. "An area (16) in r0".
the printed name of the a16r0 is "Room 0".
Understand "Area 16 in Room 0" as a16r0.
the parent of the a16r0 is r0.
the X of the a16r0 is "546852.0".
the Y of the a16r0 is "1488424.0".

[create a17r0]
a17r0 is a area. "An area (17) in r0".
the printed name of the a17r0 is "Room 0".
Understand "Area 17 in Room 0" as a17r0.
the parent of the a17r0 is r0.
the X of the a17r0 is "546812.944842407".
the Y of the a17r0 is "1488442.8194842406".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "545120.0".
the Y of the a22r0 is "1488514.0".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "545150.9153780066".
the Y of the a23r0 is "1488488.1834621995".

[create a24r0]
a24r0 is a area. "An area (24) in r0".
the printed name of the a24r0 is "Room 0".
Understand "Area 24 in Room 0" as a24r0.
the parent of the a24r0 is r0.
the X of the a24r0 is "546843.0".
the Y of the a24r0 is "1488514.0".

[create a25r0]
a25r0 is a area. "An area (25) in r0".
the printed name of the a25r0 is "Room 0".
Understand "Area 25 in Room 0" as a25r0.
the parent of the a25r0 is r0.
the X of the a25r0 is "546811.9123931624".
the Y of the a25r0 is "1488488.1662393163".

[create a65r0]
a65r0 is a area. "An area (65) in r0".
the printed name of the a65r0 is "Room 0".
Understand "Area 65 in Room 0" as a65r0.
the parent of the a65r0 is r0.
the X of the a65r0 is "546534.9973434587".
the Y of the a65r0 is "1488107.473974395".

[create a68r0]
a68r0 is a area. "An area (68) in r0".
the printed name of the a68r0 is "Room 0".
Understand "Area 68 in Room 0" as a68r0.
the parent of the a68r0 is r0.
the X of the a68r0 is "545168.092671901".
the Y of the a68r0 is "1488459.8065726862".

[create a69r0]
a69r0 is a area. "An area (69) in r0".
the printed name of the a69r0 is "Room 0".
Understand "Area 69 in Room 0" as a69r0.
the parent of the a69r0 is r0.
the X of the a69r0 is "546795.915169287".
the Y of the a69r0 is "1488461.6108476832".

[create a79r0]
a79r0 is a area. "An area (79) in r0".
the printed name of the a79r0 is "Room 0".
Understand "Area 79 in Room 0" as a79r0.
the parent of the a79r0 is r0.
the X of the a79r0 is "546039.2943851943".
the Y of the a79r0 is "1488442.3816917608".

[create a80r0]
a80r0 is a area. "An area (80) in r0".
the printed name of the a80r0 is "Room 0".
Understand "Area 80 in Room 0" as a80r0.
the parent of the a80r0 is r0.
the X of the a80r0 is "545531.1030054238".
the Y of the a80r0 is "1488330.557038129".

[create a81r0]
a81r0 is a area. "An area (81) in r0".
the printed name of the a81r0 is "Room 0".
Understand "Area 81 in Room 0" as a81r0.
the parent of the a81r0 is r0.
the X of the a81r0 is "546415.0083253963".
the Y of the a81r0 is "1488339.0902343877".

[create a86r0]
a86r0 is a area. "An area (86) in r0".
the printed name of the a86r0 is "Room 0".
Understand "Area 86 in Room 0" as a86r0.
the parent of the a86r0 is r0.
the X of the a86r0 is "546600.9241522094".
the Y of the a86r0 is "1487815.2187321647".

[create a4r1]
a4r1 is a area. "An area (4) in r1".
the printed name of the a4r1 is "Room 1".
Understand "Area 4 in Room 1" as a4r1.
the parent of the a4r1 is r1.
the X of the a4r1 is "546560.0".
the Y of the a4r1 is "1486391.0".

[create a5r1]
a5r1 is a area. "An area (5) in r1".
the printed name of the a5r1 is "Room 1".
Understand "Area 5 in Room 1" as a5r1.
the parent of the a5r1 is r1.
the X of the a5r1 is "546591.2137161085".
the Y of the a5r1 is "1486416.6339712918".

[create a6r1]
a6r1 is a area. "An area (6) in r1".
the printed name of the a6r1 is "Room 1".
Understand "Area 6 in Room 1" as a6r1.
the parent of the a6r1 is r1.
the X of the a6r1 is "546551.0".
the Y of the a6r1 is "1486480.0".

[create a7r1]
a7r1 is a area. "An area (7) in r1".
the printed name of the a7r1 is "Room 1".
Understand "Area 7 in Room 1" as a7r1.
the parent of the a7r1 is r1.
the X of the a7r1 is "546581.8730024244".
the Y of the a7r1 is "1486460.6444609193".

[create a50r1]
a50r1 is a area. "An area (50) in r1".
the printed name of the a50r1 is "Room 1".
Understand "Area 50 in Room 1" as a50r1.
the parent of the a50r1 is r1.
the X of the a50r1 is "547402.0".
the Y of the a50r1 is "1486391.0".

[create a51r1]
a51r1 is a area. "An area (51) in r1".
the printed name of the a51r1 is "Room 1".
Understand "Area 51 in Room 1" as a51r1.
the parent of the a51r1 is r1.
the X of the a51r1 is "547370.7862838916".
the Y of the a51r1 is "1486416.6339712918".

[create a54r1]
a54r1 is a area. "An area (54) in r1".
the printed name of the a54r1 is "Room 1".
Understand "Area 54 in Room 1" as a54r1.
the parent of the a54r1 is r1.
the X of the a54r1 is "547411.0".
the Y of the a54r1 is "1486480.0".

[create a55r1]
a55r1 is a area. "An area (55) in r1".
the printed name of the a55r1 is "Room 1".
Understand "Area 55 in Room 1" as a55r1.
the parent of the a55r1 is r1.
the X of the a55r1 is "547367.9383292025".
the Y of the a55r1 is "1486461.8770228894".

[create a71r1]
a71r1 is a area. "An area (71) in r1".
the printed name of the a71r1 is "Room 1".
Understand "Area 71 in Room 1" as a71r1.
the parent of the a71r1 is r1.
the X of the a71r1 is "547353.7290448343".
the Y of the a71r1 is "1486445.062703054".

[create a76r1]
a76r1 is a area. "An area (76) in r1".
the printed name of the a76r1 is "Room 1".
Understand "Area 76 in Room 1" as a76r1.
the parent of the a76r1 is r1.
the X of the a76r1 is "546605.3945333334".
the Y of the a76r1 is "1486440.2686666667".

[create a90r1]
a90r1 is a area. "An area (90) in r1".
the printed name of the a90r1 is "Room 1".
Understand "Area 90 in Room 1" as a90r1.
the parent of the a90r1 is r1.
the X of the a90r1 is "546982.1730804583".
the Y of the a90r1 is "1487259.7005024818".

[create a91r1]
a91r1 is a area. "An area (91) in r1".
the printed name of the a91r1 is "Room 1".
Understand "Area 91 in Room 1" as a91r1.
the parent of the a91r1 is r1.
the X of the a91r1 is "546987.0".
the Y of the a91r1 is "1486568.5778301887".

[create a8r2]
a8r2 is a area. "An area (8) in r2".
the printed name of the a8r2 is "Room 2".
Understand "Area 8 in Room 2" as a8r2.
the parent of the a8r2 is r2.
the X of the a8r2 is "547524.0".
the Y of the a8r2 is "1488514.0".

[create a9r2]
a9r2 is a area. "An area (9) in r2".
the printed name of the a9r2 is "Room 2".
Understand "Area 9 in Room 2" as a9r2.
the parent of the a9r2 is r2.
the X of the a9r2 is "547555.826460481".
the Y of the a9r2 is "1488505.767353952".

[create a18r2]
a18r2 is a area. "An area (18) in r2".
the printed name of the a18r2 is "Room 2".
Understand "Area 18 in Room 2" as a18r2.
the parent of the a18r2 is r2.
the X of the a18r2 is "547217.0".
the Y of the a18r2 is "1488424.0".

[create a19r2]
a19r2 is a area. "An area (19) in r2".
the printed name of the a19r2 is "Room 2".
Understand "Area 19 in Room 2" as a19r2.
the parent of the a19r2 is r2.
the X of the a19r2 is "547255.7029286583".
the Y of the a19r2 is "1488442.4774523715".

[create a20r2]
a20r2 is a area. "An area (20) in r2".
the printed name of the a20r2 is "Room 2".
Understand "Area 20 in Room 2" as a20r2.
the parent of the a20r2 is r2.
the X of the a20r2 is "547227.0".
the Y of the a20r2 is "1488514.0".

[create a21r2]
a21r2 is a area. "An area (21) in r2".
the printed name of the a21r2 is "Room 2".
Understand "Area 21 in Room 2" as a21r2.
the parent of the a21r2 is r2.
the X of the a21r2 is "547257.8949627934".
the Y of the a21r2 is "1488487.7894485786".

[create a26r2]
a26r2 is a area. "An area (26) in r2".
the printed name of the a26r2 is "Room 2".
Understand "Area 26 in Room 2" as a26r2.
the parent of the a26r2 is r2.
the X of the a26r2 is "547986.1111111111".
the Y of the a26r2 is "1488504.0".

[create a27r2]
a27r2 is a area. "An area (27) in r2".
the printed name of the a27r2 is "Room 2".
Understand "Area 27 in Room 2" as a27r2.
the parent of the a27r2 is r2.
the X of the a27r2 is "547953.1923615086".
the Y of the a27r2 is "1488505.40408955".

[create a36r2]
a36r2 is a area. "An area (36) in r2".
the printed name of the a36r2 is "Room 2".
Understand "Area 36 in Room 2" as a36r2.
the parent of the a36r2 is r2.
the X of the a36r2 is "548799.0".
the Y of the a36r2 is "1487810.0".

[create a37r2]
a37r2 is a area. "An area (37) in r2".
the printed name of the a37r2 is "Room 2".
Understand "Area 37 in Room 2" as a37r2.
the parent of the a37r2 is r2.
the X of the a37r2 is "548768.7769924189".
the Y of the a37r2 is "1487783.9929317762".

[create a38r2]
a38r2 is a area. "An area (38) in r2".
the printed name of the a38r2 is "Room 2".
Understand "Area 38 in Room 2" as a38r2.
the parent of the a38r2 is r2.
the X of the a38r2 is "548691.0".
the Y of the a38r2 is "1487822.0".

[create a39r2]
a39r2 is a area. "An area (39) in r2".
the printed name of the a39r2 is "Room 2".
Understand "Area 39 in Room 2" as a39r2.
the parent of the a39r2 is r2.
the X of the a39r2 is "548714.7621979025".
the Y of the a39r2 is "1487789.8597811218".

[create a40r2]
a40r2 is a area. "An area (40) in r2".
the printed name of the a40r2 is "Room 2".
Understand "Area 40 in Room 2" as a40r2.
the parent of the a40r2 is r2.
the X of the a40r2 is "548992.0".
the Y of the a40r2 is "1487988.0".

[create a41r2]
a41r2 is a area. "An area (41) in r2".
the printed name of the a41r2 is "Room 2".
Understand "Area 41 in Room 2" as a41r2.
the parent of the a41r2 is r2.
the X of the a41r2 is "549000.2487853912".
the Y of the a41r2 is "1488025.2932652037".

[create a42r2]
a42r2 is a area. "An area (42) in r2".
the printed name of the a42r2 is "Room 2".
Understand "Area 42 in Room 2" as a42r2.
the parent of the a42r2 is r2.
the X of the a42r2 is "549075.0".
the Y of the a42r2 is "1488025.0".

[create a43r2]
a43r2 is a area. "An area (43) in r2".
the printed name of the a43r2 is "Room 2".
Understand "Area 43 in Room 2" as a43r2.
the parent of the a43r2 is r2.
the X of the a43r2 is "549037.1025414976".
the Y of the a43r2 is "1488054.215920425".

[create a58r2]
a58r2 is a area. "An area (58) in r2".
the printed name of the a58r2 is "Room 2".
Understand "Area 58 in Room 2" as a58r2.
the parent of the a58r2 is r2.
the X of the a58r2 is "549038.0".
the Y of the a58r2 is "1488523.0".

[create a59r2]
a59r2 is a area. "An area (59) in r2".
the printed name of the a59r2 is "Room 2".
Understand "Area 59 in Room 2" as a59r2.
the parent of the a59r2 is r2.
the X of the a59r2 is "549008.0550595238".
the Y of the a59r2 is "1488500.0".

[create a63r2]
a63r2 is a area. "An area (63) in r2".
the printed name of the a63r2 is "Room 2".
Understand "Area 63 in Room 2" as a63r2.
the parent of the a63r2 is r2.
the X of the a63r2 is "548735.858357395".
the Y of the a63r2 is "1487760.9834777033".

[create a64r2]
a64r2 is a area. "An area (64) in r2".
the printed name of the a64r2 is "Room 2".
Understand "Area 64 in Room 2" as a64r2.
the parent of the a64r2 is r2.
the X of the a64r2 is "549024.2524656301".
the Y of the a64r2 is "1488058.466890702".

[create a66r2]
a66r2 is a area. "An area (66) in r2".
the printed name of the a66r2 is "Room 2".
Understand "Area 66 in Room 2" as a66r2.
the parent of the a66r2 is r2.
the X of the a66r2 is "547932.2691913696".
the Y of the a66r2 is "1488339.4267530665".

[create a67r2]
a67r2 is a area. "An area (67) in r2".
the printed name of the a67r2 is "Room 2".
Understand "Area 67 in Room 2" as a67r2.
the parent of the a67r2 is r2.
the X of the a67r2 is "547936.6849022189".
the Y of the a67r2 is "1488343.3558642007".

[create a77r2]
a77r2 is a area. "An area (77) in r2".
the printed name of the a77r2 is "Room 2".
Understand "Area 77 in Room 2" as a77r2.
the parent of the a77r2 is r2.
the X of the a77r2 is "547273.707266629".
the Y of the a77r2 is "1488461.4777749958".

[create a78r2]
a78r2 is a area. "An area (78) in r2".
the printed name of the a78r2 is "Room 2".
Understand "Area 78 in Room 2" as a78r2.
the parent of the a78r2 is r2.
the X of the a78r2 is "547564.6361867711".
the Y of the a78r2 is "1488419.4320363165".

[create a82r2]
a82r2 is a area. "An area (82) in r2".
the printed name of the a82r2 is "Room 2".
Understand "Area 82 in Room 2" as a82r2.
the parent of the a82r2 is r2.
the X of the a82r2 is "548914.626695049".
the Y of the a82r2 is "1488386.653545771".

[create a83r2]
a83r2 is a area. "An area (83) in r2".
the printed name of the a83r2 is "Room 2".
Understand "Area 83 in Room 2" as a83r2.
the parent of the a83r2 is r2.
the X of the a83r2 is "547618.0222852567".
the Y of the a83r2 is "1488378.547286384".

[create a84r2]
a84r2 is a area. "An area (84) in r2".
the printed name of the a84r2 is "Room 2".
Understand "Area 84 in Room 2" as a84r2.
the parent of the a84r2 is r2.
the X of the a84r2 is "547933.8626086956".
the Y of the a84r2 is "1487466.0".

[create a85r2]
a85r2 is a area. "An area (85) in r2".
the printed name of the a85r2 is "Room 2".
Understand "Area 85 in Room 2" as a85r2.
the parent of the a85r2 is r2.
the X of the a85r2 is "548666.7255".
the Y of the a85r2 is "1487466.0".

[create a2r3]
a2r3 is a area. "An area (2) in r3".
the printed name of the a2r3 is "Room 3".
Understand "Area 2 in Room 3" as a2r3.
the parent of the a2r3 is r3.
the X of the a2r3 is "547499.0".
the Y of the a2r3 is "1486391.0".

[create a3r3]
a3r3 is a area. "An area (3) in r3".
the printed name of the a3r3 is "Room 3".
Understand "Area 3 in Room 3" as a3r3.
the parent of the a3r3 is r3.
the X of the a3r3 is "547530.2137161086".
the Y of the a3r3 is "1486416.633971292".

[create a28r3]
a28r3 is a area. "An area (28) in r3".
the printed name of the a28r3 is "Room 3".
Understand "Area 28 in Room 3" as a28r3.
the parent of the a28r3 is r3.
the X of the a28r3 is "549087.0".
the Y of the a28r3 is "1486925.0".

[create a29r3]
a29r3 is a area. "An area (29) in r3".
the printed name of the a29r3 is "Room 3".
Understand "Area 29 in Room 3" as a29r3.
the parent of the a29r3 is r3.
the X of the a29r3 is "549093.5365348752".
the Y of the a29r3 is "1486883.8657403954".

[create a30r3]
a30r3 is a area. "An area (30) in r3".
the printed name of the a30r3 is "Room 3".
Understand "Area 30 in Room 3" as a30r3.
the parent of the a30r3 is r3.
the X of the a30r3 is "548691.0".
the Y of the a30r3 is "1487110.0".

[create a31r3]
a31r3 is a area. "An area (31) in r3".
the printed name of the a31r3 is "Room 3".
Understand "Area 31 in Room 3" as a31r3.
the parent of the a31r3 is r3.
the X of the a31r3 is "548714.0755088733".
the Y of the a31r3 is "1487142.0063361113".

[create a32r3]
a32r3 is a area. "An area (32) in r3".
the printed name of the a32r3 is "Room 3".
Understand "Area 32 in Room 3" as a32r3.
the parent of the a32r3 is r3.
the X of the a32r3 is "548799.0".
the Y of the a32r3 is "1487125.0".

[create a33r3]
a33r3 is a area. "An area (33) in r3".
the printed name of the a33r3 is "Room 3".
Understand "Area 33 in Room 3" as a33r3.
the parent of the a33r3 is r3.
the X of the a33r3 is "548767.8077975204".
the Y of the a33r3 is "1487151.433857853".

[create a46r3]
a46r3 is a area. "An area (46) in r3".
the printed name of the a46r3 is "Room 3".
Understand "Area 46 in Room 3" as a46r3.
the parent of the a46r3 is r3.
the X of the a46r3 is "547490.0".
the Y of the a46r3 is "1486480.0".

[create a47r3]
a47r3 is a area. "An area (47) in r3".
the printed name of the a47r3 is "Room 3".
Understand "Area 47 in Room 3" as a47r3.
the parent of the a47r3 is r3.
the X of the a47r3 is "547521.0059757691".
the Y of the a47r3 is "1486460.657907662".

[create a56r3]
a56r3 is a area. "An area (56) in r3".
the printed name of the a56r3 is "Room 3".
Understand "Area 56 in Room 3" as a56r3.
the parent of the a56r3 is r3.
the X of the a56r3 is "549170.0".
the Y of the a56r3 is "1486888.0".

[create a57r3]
a57r3 is a area. "An area (57) in r3".
the printed name of the a57r3 is "Room 3".
Understand "Area 57 in Room 3" as a57r3.
the parent of the a57r3 is r3.
the X of the a57r3 is "549136.1511123565".
the Y of the a57r3 is "1486867.8660087995".

[create a60r3]
a60r3 is a area. "An area (60) in r3".
the printed name of the a60r3 is "Room 3".
Understand "Area 60 in Room 3" as a60r3.
the parent of the a60r3 is r3.
the X of the a60r3 is "549097.0".
the Y of the a60r3 is "1486400.0".

[create a61r3]
a61r3 is a area. "An area (61) in r3".
the printed name of the a61r3 is "Room 3".
Understand "Area 61 in Room 3" as a61r3.
the parent of the a61r3 is r3.
the X of the a61r3 is "549058.59375".
the Y of the a61r3 is "1486434.1686912226".

[create a72r3]
a72r3 is a area. "An area (72) in r3".
the printed name of the a72r3 is "Room 3".
Understand "Area 72 in Room 3" as a72r3.
the parent of the a72r3 is r3.
the X of the a72r3 is "549112.3706921169".
the Y of the a72r3 is "1486860.504141509".

[create a73r3]
a73r3 is a area. "An area (73) in r3".
the printed name of the a73r3 is "Room 3".
Understand "Area 73 in Room 3" as a73r3.
the parent of the a73r3 is r3.
the X of the a73r3 is "548735.9241359931".
the Y of the a73r3 is "1487172.8985341636".

[create a75r3]
a75r3 is a area. "An area (75) in r3".
the printed name of the a75r3 is "Room 3".
Understand "Area 75 in Room 3" as a75r3.
the parent of the a75r3 is r3.
the X of the a75r3 is "547544.3891992549".
the Y of the a75r3 is "1486440.2597765361".

[create a87r3]
a87r3 is a area. "An area (87) in r3".
the printed name of the a87r3 is "Room 3".
Understand "Area 87 in Room 3" as a87r3.
the parent of the a87r3 is r3.
the X of the a87r3 is "547931.6643713762".
the Y of the a87r3 is "1486572.685841711".

[create a92r3]
a92r3 is a area. "An area (92) in r3".
the printed name of the a92r3 is "Room 3".
Understand "Area 92 in Room 3" as a92r3.
the parent of the a92r3 is r3.
the X of the a92r3 is "548987.1389291603".
the Y of the a92r3 is "1486513.2164822556".

[create a0r4]
a0r4 is a area. "An area (0) in r4".
the printed name of the a0r4 is "Room 4".
Understand "Area 0 in Room 4" as a0r4.
the parent of the a0r4 is r4.
the X of the a0r4 is "545098.0".
the Y of the a0r4 is "1486391.0".

[create a1r4]
a1r4 is a area. "An area (1) in r4".
the printed name of the a1r4 is "Room 4".
Understand "Area 1 in Room 4" as a1r4.
the parent of the a1r4 is r4.
the X of the a1r4 is "545129.2137161085".
the Y of the a1r4 is "1486416.633971292".

[create a34r4]
a34r4 is a area. "An area (34) in r4".
the printed name of the a34r4 is "Room 4".
Understand "Area 34 in Room 4" as a34r4.
the parent of the a34r4 is r4.
the X of the a34r4 is "546250.0".
the Y of the a34r4 is "1487260.0".

[create a35r4]
a35r4 is a area. "An area (35) in r4".
the printed name of the a35r4 is "Room 4".
Understand "Area 35 in Room 4" as a35r4.
the parent of the a35r4 is r4.
the X of the a35r4 is "546213.2".
the Y of the a35r4 is "1487352.4704098343".

[create a44r4]
a44r4 is a area. "An area (44) in r4".
the printed name of the a44r4 is "Room 4".
Understand "Area 44 in Room 4" as a44r4.
the parent of the a44r4 is r4.
the X of the a44r4 is "545089.0".
the Y of the a44r4 is "1486480.0".

[create a45r4]
a45r4 is a area. "An area (45) in r4".
the printed name of the a45r4 is "Room 4".
Understand "Area 45 in Room 4" as a45r4.
the parent of the a45r4 is r4.
the X of the a45r4 is "545120.0059757694".
the Y of the a45r4 is "1486460.657907662".

[create a48r4]
a48r4 is a area. "An area (48) in r4".
the printed name of the a48r4 is "Room 4".
Understand "Area 48 in Room 4" as a48r4.
the parent of the a48r4 is r4.
the X of the a48r4 is "545940.0".
the Y of the a48r4 is "1486391.0".

[create a49r4]
a49r4 is a area. "An area (49) in r4".
the printed name of the a49r4 is "Room 4".
Understand "Area 49 in Room 4" as a49r4.
the parent of the a49r4 is r4.
the X of the a49r4 is "545908.7862838916".
the Y of the a49r4 is "1486416.6339712918".

[create a52r4]
a52r4 is a area. "An area (52) in r4".
the printed name of the a52r4 is "Room 4".
Understand "Area 52 in Room 4" as a52r4.
the parent of the a52r4 is r4.
the X of the a52r4 is "545949.0".
the Y of the a52r4 is "1486480.0".

[create a53r4]
a53r4 is a area. "An area (53) in r4".
the printed name of the a53r4 is "Room 4".
Understand "Area 53 in Room 4" as a53r4.
the parent of the a53r4 is r4.
the X of the a53r4 is "545906.1485252036".
the Y of the a53r4 is "1486461.8557671143".

[create a62r4]
a62r4 is a area. "An area (62) in r4".
the printed name of the a62r4 is "Room 4".
Understand "Area 62 in Room 4" as a62r4.
the parent of the a62r4 is r4.
the X of the a62r4 is "546212.3646166535".
the Y of the a62r4 is "1487354.5764807123".

[create a70r4]
a70r4 is a area. "An area (70) in r4".
the printed name of the a70r4 is "Room 4".
Understand "Area 70 in Room 4" as a70r4.
the parent of the a70r4 is r4.
the X of the a70r4 is "545891.7817693062".
the Y of the a70r4 is "1486444.9748289345".

[create a74r4]
a74r4 is a area. "An area (74) in r4".
the printed name of the a74r4 is "Room 4".
Understand "Area 74 in Room 4" as a74r4.
the parent of the a74r4 is r4.
the X of the a74r4 is "545143.389199255".
the Y of the a74r4 is "1486440.2597765361".

[create a88r4]
a88r4 is a area. "An area (88) in r4".
the printed name of the a88r4 is "Room 4".
Understand "Area 88 in Room 4" as a88r4.
the parent of the a88r4 is r4.
the X of the a88r4 is "545531.7860830325".
the Y of the a88r4 is "1487338.5".

[create a89r4]
a89r4 is a area. "An area (89) in r4".
the printed name of the a89r4 is "Room 4".
Understand "Area 89 in Room 4" as a89r4.
the parent of the a89r4 is r4.
the X of the a89r4 is "545525.0".
the Y of the a89r4 is "1486568.5778301887".

[create d0]
d0 is a door. "A door between a84r2 and a87r3".
d0 is south of a84r2 and north of a87r3.

[create d1]
d1 is a door. "A door between a85r2 and a73r3".
d1 is south of a85r2 and north of a73r3.

[create d4]
d4 is a door. "A door between a90r1 and a83r2".
d4 is northeast of a90r1 and southwest of a83r2.

[create d6]
d6 is a door. "A door between a86r0 and a90r1".
d6 is southeast of a86r0 and northwest of a90r1.

[create d8]
d8 is a door. "A door between a86r0 and a62r4".
d8 is southwest of a86r0 and northeast of a62r4.

[create d9]
d9 is a door. "A door between a80r0 and a88r4".
d9 is south of a80r0 and north of a88r4.

northeast of a0r4 is southwest of a1r4.

northeast of a2r3 is southwest of a3r3.

northeast of a4r1 is southwest of a5r1.

southeast of a6r1 is northwest of a7r1.

east of a8r2 is west of a9r2.

east of a10r0 is west of a11r0.

west of a12r0 is east of a13r0.

northeast of a14r0 is southwest of a15r0.

northwest of a16r0 is southeast of a17r0.

northeast of a18r2 is southwest of a19r2.

southeast of a20r2 is northwest of a21r2.

southeast of a22r0 is northwest of a23r0.

southwest of a24r0 is northeast of a25r0.

west of a26r2 is east of a27r2.

south of a28r3 is north of a29r3.

northeast of a30r3 is southwest of a31r3.

northwest of a32r3 is southeast of a33r3.

north of a34r4 is south of a35r4.

southwest of a36r2 is northeast of a37r2.

southeast of a38r2 is northwest of a39r2.

north of a40r2 is south of a41r2.

northwest of a42r2 is southeast of a43r2.

southeast of a44r4 is northwest of a45r4.

southeast of a46r3 is northwest of a47r3.

northwest of a48r4 is southeast of a49r4.

northwest of a50r1 is southeast of a51r1.

southwest of a52r4 is northeast of a53r4.

southwest of a54r1 is northeast of a55r1.

southwest of a56r3 is northeast of a57r3.

southwest of a58r2 is northeast of a59r2.

northwest of a60r3 is southeast of a61r3.

north of a35r4 is south of a62r4.

southwest of a37r2 is northeast of a63r2.

southeast of a39r2 is northwest of a63r2.

northeast of a41r2 is southwest of a64r2.

west of a43r2 is east of a64r2.

southwest of a13r0 is northeast of a65r0.

northeast of a66r2 is southwest of a67r2.

northeast of a15r0 is southwest of a68r0.

northwest of a17r0 is southeast of a69r0.

northwest of a49r4 is southeast of a70r4.

northwest of a51r1 is southeast of a71r1.

southwest of a53r4 is northeast of a70r4.

southwest of a55r1 is northeast of a71r1.

west of a57r3 is east of a72r3.

southeast of a29r3 is northwest of a72r3.

northeast of a31r3 is southwest of a73r3.

northwest of a33r3 is southeast of a73r3.

northeast of a1r4 is southwest of a74r4.

northeast of a3r3 is southwest of a75r3.

northeast of a5r1 is southwest of a76r1.

southeast of a7r1 is northwest of a76r1.

southeast of a45r4 is northwest of a74r4.

southeast of a47r3 is northwest of a75r3.

northeast of a19r2 is southwest of a77r2.

southeast of a21r2 is northwest of a77r2.

southeast of a23r0 is northwest of a68r0.

southwest of a25r0 is northeast of a69r0.

south of a27r2 is north of a67r2.

south of a9r2 is north of a78r2.

south of a11r0 is north of a79r0.

east of a68r0 is west of a80r0.

east of a77r2 is west of a78r2.

east of a79r0 is west of a81r0.

east of a67r2 is west of a82r2.

west of a66r2 is east of a83r2.

southeast of a78r2 is northwest of a83r2.

southwest of a59r2 is northeast of a82r2.

east of a84r2 is west of a85r2.

south of a63r2 is north of a85r2.

south of a66r2 is north of a84r2.

south of a65r0 is north of a86r0.

north of a64r2 is south of a82r2.

northwest of a65r0 is southeast of a81r0.

west of a69r0 is east of a81r0.

west of a79r0 is east of a80r0.

south of a88r4 is north of a89r4.

west of a62r4 is east of a88r4.

west of a70r4 is east of a89r4.

east of a76r1 is west of a91r1.

east of a74r4 is west of a89r4.

east of a75r3 is west of a87r3.

east of a87r3 is west of a92r3.

northwest of a61r3 is southeast of a92r3.

south of a72r3 is north of a92r3.

south of a90r1 is north of a91r1.

west of a71r1 is east of a91r1.

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
the player is in a40r2.

the orientation of the player is 6.
