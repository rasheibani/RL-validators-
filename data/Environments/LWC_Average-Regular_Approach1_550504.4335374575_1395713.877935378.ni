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

[create a0r0]
a0r0 is a area. "An area (0) in r0".
the printed name of the a0r0 is "Room 0".
Understand "Area 0 in Room 0" as a0r0.
the parent of the a0r0 is r0.
the X of the a0r0 is "546846.2031523075".
the Y of the a0r0 is "1395216.2575374735".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "546948.6223771108".
the Y of the a1r0 is "1395364.0579879202".

[create a4r0]
a4r0 is a area. "An area (4) in r0".
the printed name of the a4r0 is "Room 0".
Understand "Area 4 in Room 0" as a4r0.
the parent of the a4r0 is r0.
the X of the a4r0 is "546561.5705443528".
the Y of the a4r0 is "1395312.939656898".

[create a5r0]
a5r0 is a area. "An area (5) in r0".
the printed name of the a5r0 is "Room 0".
Understand "Area 5 in Room 0" as a5r0.
the parent of the a5r0 is r0.
the X of the a5r0 is "546718.1319370695".
the Y of the a5r0 is "1395656.3984667852".

[create a8r0]
a8r0 is a area. "An area (8) in r0".
the printed name of the a8r0 is "Room 0".
Understand "Area 8 in Room 0" as a8r0.
the parent of the a8r0 is r0.
the X of the a8r0 is "546962.7496886395".
the Y of the a8r0 is "1395362.3173736061".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "547056.2968536067".
the Y of the a9r0 is "1395251.4209743899".

[create a16r0]
a16r0 is a area. "An area (16) in r0".
the printed name of the a16r0 is "Room 0".
Understand "Area 16 in Room 0" as a16r0.
the parent of the a16r0 is r0.
the X of the a16r0 is "545140.389199255".
the Y of the a16r0 is "1395240.2597765364".

[create a17r0]
a17r0 is a area. "An area (17) in r0".
the printed name of the a17r0 is "Room 0".
Understand "Area 17 in Room 0" as a17r0.
the parent of the a17r0 is r0.
the X of the a17r0 is "545522.0938967268".
the Y of the a17r0 is "1395368.0255155074".

[create a18r0]
a18r0 is a area. "An area (18) in r0".
the printed name of the a18r0 is "Room 0".
Understand "Area 18 in Room 0" as a18r0.
the parent of the a18r0 is r0.
the X of the a18r0 is "546633.28125".
the Y of the a18r0 is "1395234.2597374606".

[create a36r0]
a36r0 is a area. "An area (36) in r0".
the printed name of the a36r0 is "Room 0".
Understand "Area 36 in Room 0" as a36r0.
the parent of the a36r0 is r0.
the X of the a36r0 is "546814.9851113801".
the Y of the a36r0 is "1395681.6710919573".

[create a37r0]
a37r0 is a area. "An area (37) in r0".
the printed name of the a37r0 is "Room 0".
Understand "Area 37 in Room 0" as a37r0.
the parent of the a37r0 is r0.
the X of the a37r0 is "546792.9850062027".
the Y of the a37r0 is "1395792.9077627996".

[create a76r0]
a76r0 is a area. "An area (76) in r0".
the printed name of the a76r0 is "Room 0".
Understand "Area 76 in Room 0" as a76r0.
the parent of the a76r0 is r0.
the X of the a76r0 is "546671.8469781214".
the Y of the a76r0 is "1395720.610383247".

[create a77r0]
a77r0 is a area. "An area (77) in r0".
the printed name of the a77r0 is "Room 0".
Understand "Area 77 in Room 0" as a77r0.
the parent of the a77r0 is r0.
the X of the a77r0 is "546651.973715612".
the Y of the a77r0 is "1395689.2692133465".

[create a82r0]
a82r0 is a area. "An area (82) in r0".
the printed name of the a82r0 is "Room 0".
Understand "Area 82 in Room 0" as a82r0.
the parent of the a82r0 is r0.
the X of the a82r0 is "546685.6937489837".
the Y of the a82r0 is "1395764.9114503693".

[create a87r0]
a87r0 is a area. "An area (87) in r0".
the printed name of the a87r0 is "Room 0".
Understand "Area 87 in Room 0" as a87r0.
the parent of the a87r0 is r0.
the X of the a87r0 is "545095.0".
the Y of the a87r0 is "1395191.0".

[create a88r0]
a88r0 is a area. "An area (88) in r0".
the printed name of the a88r0 is "Room 0".
Understand "Area 88 in Room 0" as a88r0.
the parent of the a88r0 is r0.
the X of the a88r0 is "545126.2137161085".
the Y of the a88r0 is "1395216.6339712918".

[create a89r0]
a89r0 is a area. "An area (89) in r0".
the printed name of the a89r0 is "Room 0".
Understand "Area 89 in Room 0" as a89r0.
the parent of the a89r0 is r0.
the X of the a89r0 is "545086.0".
the Y of the a89r0 is "1395280.0".

[create a90r0]
a90r0 is a area. "An area (90) in r0".
the printed name of the a90r0 is "Room 0".
Understand "Area 90 in Room 0" as a90r0.
the parent of the a90r0 is r0.
the X of the a90r0 is "545117.0059757691".
the Y of the a90r0 is "1395260.657907662".

[create a92r0]
a92r0 is a area. "An area (92) in r0".
the printed name of the a92r0 is "Room 0".
Understand "Area 92 in Room 0" as a92r0.
the parent of the a92r0 is r0.
the X of the a92r0 is "547097.0".
the Y of the a92r0 is "1395200.0".

[create a96r0]
a96r0 is a area. "An area (96) in r0".
the printed name of the a96r0 is "Room 0".
Understand "Area 96 in Room 0" as a96r0.
the parent of the a96r0 is r0.
the X of the a96r0 is "546671.0".
the Y of the a96r0 is "1395200.0".

[create a97r0]
a97r0 is a area. "An area (97) in r0".
the printed name of the a97r0 is "Room 0".
Understand "Area 97 in Room 0" as a97r0.
the parent of the a97r0 is r0.
the X of the a97r0 is "546815.0".
the Y of the a97r0 is "1395172.0".

[create a15r1]
a15r1 is a area. "An area (15) in r1".
the printed name of the a15r1 is "Room 1".
Understand "Area 15 in Room 1" as a15r1.
the parent of the a15r1 is r1.
the X of the a15r1 is "547608.8708448359".
the Y of the a15r1 is "1397005.3162913683".

[create a21r1]
a21r1 is a area. "An area (21) in r1".
the printed name of the a21r1 is "Room 1".
Understand "Area 21 in Room 1" as a21r1.
the parent of the a21r1 is r1.
the X of the a21r1 is "547536.333333333".
the Y of the a21r1 is "1397260.9694444446".

[create a22r1]
a22r1 is a area. "An area (22) in r1".
the printed name of the a22r1 is "Room 1".
Understand "Area 22 in Room 1" as a22r1.
the parent of the a22r1 is r1.
the X of the a22r1 is "547603.5167685037".
the Y of the a22r1 is "1397173.5133957588".

[create a41r1]
a41r1 is a area. "An area (41) in r1".
the printed name of the a41r1 is "Room 1".
Understand "Area 41 in Room 1" as a41r1.
the parent of the a41r1 is r1.
the X of the a41r1 is "547703.0".
the Y of the a41r1 is "1397304.0".

[create a42r1]
a42r1 is a area. "An area (42) in r1".
the printed name of the a42r1 is "Room 1".
Understand "Area 42 in Room 1" as a42r1.
the parent of the a42r1 is r1.
the X of the a42r1 is "547669.6666666665".
the Y of the a42r1 is "1397253.5364495006".

[create a95r1]
a95r1 is a area. "An area (95) in r1".
the printed name of the a95r1 is "Room 1".
Understand "Area 95 in Room 1" as a95r1.
the parent of the a95r1 is r1.
the X of the a95r1 is "547503.0".
the Y of the a95r1 is "1397304.0".

[create a2r2]
a2r2 is a area. "An area (2) in r2".
the printed name of the a2r2 is "Room 2".
Understand "Area 2 in Room 2" as a2r2.
the parent of the a2r2 is r2.
the X of the a2r2 is "548107.7043713275".
the Y of the a2r2 is "1395222.7067112252".

[create a3r2]
a3r2 is a area. "An area (3) in r2".
the printed name of the a3r2 is "Room 2".
Understand "Area 3 in Room 2" as a3r2.
the parent of the a3r2 is r2.
the X of the a3r2 is "548203.2390740268".
the Y of the a3r2 is "1395362.431282769".

[create a10r2]
a10r2 is a area. "An area (10) in r2".
the printed name of the a10r2 is "Room 2".
Understand "Area 10 in Room 2" as a10r2.
the parent of the a10r2 is r2.
the X of the a10r2 is "548225.2896516463".
the Y of the a10r2 is "1395360.8833195835".

[create a11r2]
a11r2 is a area. "An area (11) in r2".
the printed name of the a11r2 is "Room 2".
Understand "Area 11 in Room 2" as a11r2.
the parent of the a11r2 is r2.
the X of the a11r2 is "548319.4539705468".
the Y of the a11r2 is "1395248.7643334025".

[create a91r2]
a91r2 is a area. "An area (91) in r2".
the printed name of the a91r2 is "Room 2".
Understand "Area 91 in Room 2" as a91r2.
the parent of the a91r2 is r2.
the X of the a91r2 is "548077.0".
the Y of the a91r2 is "1395172.0".

[create a93r2]
a93r2 is a area. "An area (93) in r2".
the printed name of the a93r2 is "Room 2".
Understand "Area 93 in Room 2" as a93r2.
the parent of the a93r2 is r2.
the X of the a93r2 is "548360.0".
the Y of the a93r2 is "1395200.0".

[create a14r3]
a14r3 is a area. "An area (14) in r3".
the printed name of the a14r3 is "Room 3".
Understand "Area 14 in Room 3" as a14r3.
the parent of the a14r3 is r3.
the X of the a14r3 is "548732.1200470268".
the Y of the a14r3 is "1396594.3390806045".

[create a26r3]
a26r3 is a area. "An area (26) in r3".
the printed name of the a26r3 is "Room 3".
Understand "Area 26 in Room 3" as a26r3.
the parent of the a26r3 is r3.
the X of the a26r3 is "548401.4921500897".
the Y of the a26r3 is "1397260.7986347189".

[create a27r3]
a27r3 is a area. "An area (27) in r3".
the printed name of the a27r3 is "Room 3".
Understand "Area 27 in Room 3" as a27r3.
the parent of the a27r3 is r3.
the X of the a27r3 is "548735.9486065779".
the Y of the a27r3 is "1397178.1064398694".

[create a28r3]
a28r3 is a area. "An area (28) in r3".
the printed name of the a28r3 is "Room 3".
Understand "Area 28 in Room 3" as a28r3.
the parent of the a28r3 is r3.
the X of the a28r3 is "549065.5509290381".
the Y of the a28r3 is "1397261.006208871".

[create a29r3]
a29r3 is a area. "An area (29) in r3".
the printed name of the a29r3 is "Room 3".
Understand "Area 29 in Room 3" as a29r3.
the parent of the a29r3 is r3.
the X of the a29r3 is "548787.3007254268".
the Y of the a29r3 is "1396580.654272361".

[create a30r3]
a30r3 is a area. "An area (30) in r3".
the printed name of the a30r3 is "Room 3".
Understand "Area 30 in Room 3" as a30r3.
the parent of the a30r3 is r3.
the X of the a30r3 is "549222.0090168878".
the Y of the a30r3 is "1397123.9658582641".

[create a31r3]
a31r3 is a area. "An area (31) in r3".
the printed name of the a31r3 is "Room 3".
Understand "Area 31 in Room 3" as a31r3.
the parent of the a31r3 is r3.
the X of the a31r3 is "550376.1267511034".
the Y of the a31r3 is "1397200.9824409904".

[create a32r3]
a32r3 is a area. "An area (32) in r3".
the printed name of the a32r3 is "Room 3".
Understand "Area 32 in Room 3" as a32r3.
the parent of the a32r3 is r3.
the X of the a32r3 is "550446.2035569828".
the Y of the a32r3 is "1396858.4830703116".

[create a33r3]
a33r3 is a area. "An area (33) in r3".
the printed name of the a33r3 is "Room 3".
Understand "Area 33 in Room 3" as a33r3.
the parent of the a33r3 is r3.
the X of the a33r3 is "549311.0120769149".
the Y of the a33r3 is "1397214.7204307124".

[create a34r3]
a34r3 is a area. "An area (34) in r3".
the printed name of the a34r3 is "Room 3".
Understand "Area 34 in Room 3" as a34r3.
the parent of the a34r3 is r3.
the X of the a34r3 is "550426.9520547944".
the Y of the a34r3 is "1397281.0".

[create a38r3]
a38r3 is a area. "An area (38) in r3".
the printed name of the a38r3 is "Room 3".
Understand "Area 38 in Room 3" as a38r3.
the parent of the a38r3 is r3.
the X of the a38r3 is "548835.0473480653".
the Y of the a38r3 is "1396350.987455197".

[create a47r3]
a47r3 is a area. "An area (47) in r3".
the printed name of the a47r3 is "Room 3".
Understand "Area 47 in Room 3" as a47r3.
the parent of the a47r3 is r3.
the X of the a47r3 is "548354.0".
the Y of the a47r3 is "1397314.0".

[create a48r3]
a48r3 is a area. "An area (48) in r3".
the printed name of the a48r3 is "Room 3".
Understand "Area 48 in Room 3" as a48r3.
the parent of the a48r3 is r3.
the X of the a48r3 is "548384.915378007".
the Y of the a48r3 is "1397288.1834621993".

[create a53r3]
a53r3 is a area. "An area (53) in r3".
the printed name of the a53r3 is "Room 3".
Understand "Area 53 in Room 3" as a53r3.
the parent of the a53r3 is r3.
the X of the a53r3 is "549113.0".
the Y of the a53r3 is "1397314.0".

[create a54r3]
a54r3 is a area. "An area (54) in r3".
the printed name of the a54r3 is "Room 3".
Understand "Area 54 in Room 3" as a54r3.
the parent of the a54r3 is r3.
the X of the a54r3 is "549081.9123931623".
the Y of the a54r3 is "1397288.1662393163".

[create a55r3]
a55r3 is a area. "An area (55) in r3".
the printed name of the a55r3 is "Room 3".
Understand "Area 55 in Room 3" as a55r3.
the parent of the a55r3 is r3.
the X of the a55r3 is "549259.0".
the Y of the a55r3 is "1397332.0".

[create a56r3]
a56r3 is a area. "An area (56) in r3".
the printed name of the a56r3 is "Room 3".
Understand "Area 56 in Room 3" as a56r3.
the parent of the a56r3 is r3.
the X of the a56r3 is "549295.2142857141".
the Y of the a56r3 is "1397250.7207578144".

[create a57r3]
a57r3 is a area. "An area (57) in r3".
the printed name of the a57r3 is "Room 3".
Understand "Area 57 in Room 3" as a57r3.
the parent of the a57r3 is r3.
the X of the a57r3 is "550414.0".
the Y of the a57r3 is "1396788.0".

[create a58r3]
a58r3 is a area. "An area (58) in r3".
the printed name of the a58r3 is "Room 3".
Understand "Area 58 in Room 3" as a58r3.
the parent of the a58r3 is r3.
the X of the a58r3 is "550421.4890701754".
the Y of the a58r3 is "1396826.997491228".

[create a59r3]
a59r3 is a area. "An area (59) in r3".
the printed name of the a59r3 is "Room 3".
Understand "Area 59 in Room 3" as a59r3.
the parent of the a59r3 is r3.
the X of the a59r3 is "550497.0".
the Y of the a59r3 is "1396825.0".

[create a60r3]
a60r3 is a area. "An area (60) in r3".
the printed name of the a60r3 is "Room 3".
Understand "Area 60 in Room 3" as a60r3.
the parent of the a60r3 is r3.
the X of the a60r3 is "550459.1025414974".
the Y of the a60r3 is "1396854.215920425".

[create a61r3]
a61r3 is a area. "An area (61) in r3".
the printed name of the a61r3 is "Room 3".
Understand "Area 61 in Room 3" as a61r3.
the parent of the a61r3 is r3.
the X of the a61r3 is "549122.0".
the Y of the a61r3 is "1397224.0".

[create a62r3]
a62r3 is a area. "An area (62) in r3".
the printed name of the a62r3 is "Room 3".
Understand "Area 62 in Room 3" as a62r3.
the parent of the a62r3 is r3.
the X of the a62r3 is "549144.0084243885".
the Y of the a62r3 is "1397194.9196682633".

[create a67r3]
a67r3 is a area. "An area (67) in r3".
the printed name of the a67r3 is "Room 3".
Understand "Area 67 in Room 3" as a67r3.
the parent of the a67r3 is r3.
the X of the a67r3 is "548345.0".
the Y of the a67r3 is "1397224.0".

[create a68r3]
a68r3 is a area. "An area (68) in r3".
the printed name of the a68r3 is "Room 3".
Understand "Area 68 in Room 3" as a68r3.
the parent of the a68r3 is r3.
the X of the a68r3 is "548385.7622699386".
the Y of the a68r3 is "1397242.6487730062".

[create a69r3]
a69r3 is a area. "An area (69) in r3".
the printed name of the a69r3 is "Room 3".
Understand "Area 69 in Room 3" as a69r3.
the parent of the a69r3 is r3.
the X of the a69r3 is "549081.6625777471".
the Y of the a69r3 is "1397242.6912577746".

[create a85r3]
a85r3 is a area. "An area (85) in r3".
the printed name of the a85r3 is "Room 3".
Understand "Area 85 in Room 3" as a85r3.
the parent of the a85r3 is r3.
the X of the a85r3 is "548675.0".
the Y of the a85r3 is "1396388.5161290322".

[create a86r3]
a86r3 is a area. "An area (86) in r3".
the printed name of the a86r3 is "Room 3".
Understand "Area 86 in Room 3" as a86r3.
the parent of the a86r3 is r3.
the X of the a86r3 is "548675.3081281073".
the Y of the a86r3 is "1396350.9874551971".

[create a94r3]
a94r3 is a area. "An area (94) in r3".
the printed name of the a94r3 is "Room 3".
Understand "Area 94 in Room 3" as a94r3.
the parent of the a94r3 is r3.
the X of the a94r3 is "550387.0".
the Y of the a94r3 is "1397286.0".

[create a6r4]
a6r4 is a area. "An area (6) in r4".
the printed name of the a6r4 is "Room 4".
Understand "Area 6 in Room 4" as a6r4.
the parent of the a6r4 is r4.
the X of the a6r4 is "548872.3838838992".
the Y of the a6r4 is "1395798.68330402".

[create a7r4]
a7r4 is a area. "An area (7) in r4".
the printed name of the a7r4 is "Room 4".
Understand "Area 7 in Room 4" as a7r4.
the parent of the a7r4 is r4.
the X of the a7r4 is "549318.3005528575".
the Y of the a7r4 is "1395298.8583478578".

[create a12r4]
a12r4 is a area. "An area (12) in r4".
the printed name of the a12r4 is "Room 4".
Understand "Area 12 in Room 4" as a12r4.
the parent of the a12r4 is r4.
the X of the a12r4 is "550399.7849280776".
the Y of the a12r4 is "1395390.0136747023".

[create a13r4]
a13r4 is a area. "An area (13) in r4".
the printed name of the a13r4 is "Room 4".
Understand "Area 13 in Room 4" as a13r4.
the parent of the a13r4 is r4.
the X of the a13r4 is "550524.2748579546".
the Y of the a13r4 is "1395688.9249526516".

[create a72r4]
a72r4 is a area. "An area (72) in r4".
the printed name of the a72r4 is "Room 4".
Understand "Area 72 in Room 4" as a72r4.
the parent of the a72r4 is r4.
the X of the a72r4 is "549265.0".
the Y of the a72r4 is "1395163.0".

[create a73r4]
a73r4 is a area. "An area (73) in r4".
the printed name of the a73r4 is "Room 4".
Understand "Area 73 in Room 4" as a73r4.
the parent of the a73r4 is r4.
the X of the a73r4 is "549301.0".
the Y of the a73r4 is "1395255.3867768596".

[create a74r4]
a74r4 is a area. "An area (74) in r4".
the printed name of the a74r4 is "Room 4".
Understand "Area 74 in Room 4" as a74r4.
the parent of the a74r4 is r4.
the X of the a74r4 is "550509.0".
the Y of the a74r4 is "1395304.0".

[create a75r4]
a75r4 is a area. "An area (75) in r4".
the printed name of the a75r4 is "Room 4".
Understand "Area 75 in Room 4" as a75r4.
the parent of the a75r4 is r4.
the X of the a75r4 is "550456.4502168536".
the Y of the a75r4 is "1395348.7548575376".

[create a78r4]
a78r4 is a area. "An area (78) in r4".
the printed name of the a78r4 is "Room 4".
Understand "Area 78 in Room 4" as a78r4.
the parent of the a78r4 is r4.
the X of the a78r4 is "550580.0".
the Y of the a78r4 is "1395716.0".

[create a79r4]
a79r4 is a area. "An area (79) in r4".
the printed name of the a79r4 is "Room 4".
Understand "Area 79 in Room 4" as a79r4.
the parent of the a79r4 is r4.
the X of the a79r4 is "550546.3349056605".
the Y of the a79r4 is "1395696.2783018867".

[create a80r4]
a80r4 is a area. "An area (80) in r4".
the printed name of the a80r4 is "Room 4".
Understand "Area 80 in Room 4" as a80r4.
the parent of the a80r4 is r4.
the X of the a80r4 is "550497.0".
the Y of the a80r4 is "1395753.0".

[create a81r4]
a81r4 is a area. "An area (81) in r4".
the printed name of the a81r4 is "Room 4".
Understand "Area 81 in Room 4" as a81r4.
the parent of the a81r4 is r4.
the X of the a81r4 is "550504.4335374575".
the Y of the a81r4 is "1395713.877935378".

[create a83r4]
a83r4 is a area. "An area (83) in r4".
the printed name of the a83r4 is "Room 4".
Understand "Area 83 in Room 4" as a83r4.
the parent of the a83r4 is r4.
the X of the a83r4 is "548675.0".
the Y of the a83r4 is "1395713.0".

[create a84r4]
a84r4 is a area. "An area (84) in r4".
the printed name of the a84r4 is "Room 4".
Understand "Area 84 in Room 4" as a84r4.
the parent of the a84r4 is r4.
the X of the a84r4 is "548760.1076235287".
the Y of the a84r4 is "1395750.528673835".

[create a19r5]
a19r5 is a area. "An area (19) in r5".
the printed name of the a19r5 is "Room 5".
Understand "Area 19 in Room 5" as a19r5.
the parent of the a19r5 is r5.
the X of the a19r5 is "545525.0".
the Y of the a19r5 is "1397135.3372641508".

[create a20r5]
a20r5 is a area. "An area (20) in r5".
the printed name of the a20r5 is "Room 5".
Understand "Area 20 in Room 5" as a20r5.
the parent of the a20r5 is r5.
the X of the a20r5 is "545882.8606168446".
the Y of the a20r5 is "1397259.8602906286".

[create a35r5]
a35r5 is a area. "An area (35) in r5".
the printed name of the a35r5 is "Room 5".
Understand "Area 35 in Room 5" as a35r5.
the parent of the a35r5 is r5.
the X of the a35r5 is "545168.0926719012".
the Y of the a35r5 is "1397259.8065726864".

[create a45r5]
a45r5 is a area. "An area (45) in r5".
the printed name of the a45r5 is "Room 5".
Understand "Area 45 in Room 5" as a45r5.
the parent of the a45r5 is r5.
the X of the a45r5 is "545120.0".
the Y of the a45r5 is "1397314.0".

[create a46r5]
a46r5 is a area. "An area (46) in r5".
the printed name of the a46r5 is "Room 5".
Understand "Area 46 in Room 5" as a46r5.
the parent of the a46r5 is r5.
the X of the a46r5 is "545150.9153780071".
the Y of the a46r5 is "1397288.1834621993".

[create a51r5]
a51r5 is a area. "An area (51) in r5".
the printed name of the a51r5 is "Room 5".
Understand "Area 51 in Room 5" as a51r5.
the parent of the a51r5 is r5.
the X of the a51r5 is "545931.0".
the Y of the a51r5 is "1397314.0".

[create a52r5]
a52r5 is a area. "An area (52) in r5".
the printed name of the a52r5 is "Room 5".
Understand "Area 52 in Room 5" as a52r5.
the parent of the a52r5 is r5.
the X of the a52r5 is "545899.9123931623".
the Y of the a52r5 is "1397288.1662393163".

[create a63r5]
a63r5 is a area. "An area (63) in r5".
the printed name of the a63r5 is "Room 5".
Understand "Area 63 in Room 5" as a63r5.
the parent of the a63r5 is r5.
the X of the a63r5 is "545111.0".
the Y of the a63r5 is "1397224.0".

[create a64r5]
a64r5 is a area. "An area (64) in r5".
the printed name of the a64r5 is "Room 5".
Understand "Area 64 in Room 5" as a64r5.
the parent of the a64r5 is r5.
the X of the a64r5 is "545153.0599999999".
the Y of the a64r5 is "1397242.5189999999".

[create a65r5]
a65r5 is a area. "An area (65) in r5".
the printed name of the a65r5 is "Room 5".
Understand "Area 65 in Room 5" as a65r5.
the parent of the a65r5 is r5.
the X of the a65r5 is "545940.0".
the Y of the a65r5 is "1397224.0".

[create a66r5]
a66r5 is a area. "An area (66) in r5".
the printed name of the a66r5 is "Room 5".
Understand "Area 66 in Room 5" as a66r5.
the parent of the a66r5 is r5.
the X of the a66r5 is "545897.9400000001".
the Y of the a66r5 is "1397242.519".

[create a23r6]
a23r6 is a area. "An area (23) in r6".
the printed name of the a23r6 is "Room 6".
Understand "Area 23 in Room 6" as a23r6.
the parent of the a23r6 is r6.
the X of the a23r6 is "546109.281371821".
the Y of the a23r6 is "1397261.2845894436".

[create a24r6]
a24r6 is a area. "An area (24) in r6".
the printed name of the a24r6 is "Room 6".
Understand "Area 24 in Room 6" as a24r6.
the parent of the a24r6 is r6.
the X of the a24r6 is "546464.79770735".
the Y of the a24r6 is "1397144.5677680378".

[create a25r6]
a25r6 is a area. "An area (25) in r6".
the printed name of the a25r6 is "Room 6".
Understand "Area 25 in Room 6" as a25r6.
the parent of the a25r6 is r6.
the X of the a25r6 is "546838.1653756483".
the Y of the a25r6 is "1397261.2658517454".

[create a39r6]
a39r6 is a area. "An area (39) in r6".
the printed name of the a39r6 is "Room 6".
Understand "Area 39 in Room 6" as a39r6.
the parent of the a39r6 is r6.
the X of the a39r6 is "546895.0".
the Y of the a39r6 is "1397224.0".

[create a40r6]
a40r6 is a area. "An area (40) in r6".
the printed name of the a40r6 is "Room 6".
Understand "Area 40 in Room 6" as a40r6.
the parent of the a40r6 is r6.
the X of the a40r6 is "546854.6686930091".
the Y of the a40r6 is "1397242.2965214455".

[create a43r6]
a43r6 is a area. "An area (43) in r6".
the printed name of the a43r6 is "Room 6".
Understand "Area 43 in Room 6" as a43r6.
the parent of the a43r6 is r6.
the X of the a43r6 is "546885.0".
the Y of the a43r6 is "1397314.0".

[create a44r6]
a44r6 is a area. "An area (44) in r6".
the printed name of the a44r6 is "Room 6".
Understand "Area 44 in Room 6" as a44r6.
the parent of the a44r6 is r6.
the X of the a44r6 is "546854.1050372066".
the Y of the a44r6 is "1397287.7894485786".

[create a49r6]
a49r6 is a area. "An area (49) in r6".
the printed name of the a49r6 is "Room 6".
Understand "Area 49 in Room 6" as a49r6.
the parent of the a49r6 is r6.
the X of the a49r6 is "546062.0".
the Y of the a49r6 is "1397314.0".

[create a50r6]
a50r6 is a area. "An area (50) in r6".
the printed name of the a50r6 is "Room 6".
Understand "Area 50 in Room 6" as a50r6.
the parent of the a50r6 is r6.
the X of the a50r6 is "546093.0876068375".
the Y of the a50r6 is "1397288.1662393163".

[create a70r6]
a70r6 is a area. "An area (70) in r6".
the printed name of the a70r6 is "Room 6".
Understand "Area 70 in Room 6" as a70r6.
the parent of the a70r6 is r6.
the X of the a70r6 is "546053.0".
the Y of the a70r6 is "1397224.0".

[create a71r6]
a71r6 is a area. "An area (71) in r6".
the printed name of the a71r6 is "Room 6".
Understand "Area 71 in Room 6" as a71r6.
the parent of the a71r6 is r6.
the X of the a71r6 is "546092.4941364606".
the Y of the a71r6 is "1397242.7755863538".

[create d0]
d0 is a door. "A door between a10r2 and a14r3".
d0 is north of a10r2 and south of a14r3.

[create d2]
d2 is a door. "A door between a8r0 and a15r1".
d2 is north of a8r0 and south of a15r1.

[create d4]
d4 is a door. "A door between a17r0 and a19r5".
d4 is north of a17r0 and south of a19r5.

[create d6]
d6 is a door. "A door between a15r1 and a3r2".
d6 is southeast of a15r1 and northwest of a3r2.

[create d8]
d8 is a door. "A door between a37r0 and a24r6".
d8 is north of a37r0 and south of a24r6.

[create d10]
d10 is a door. "A door between a38r3 and a6r4".
d10 is south of a38r3 and north of a6r4.

northeast of a0r0 is southwest of a1r0.

northeast of a2r2 is southwest of a3r2.

northeast of a4r0 is southwest of a5r0.

southeast of a6r4 is northwest of a7r4.

southeast of a8r0 is northwest of a9r0.

southeast of a10r2 is northwest of a11r2.

northeast of a12r4 is southwest of a13r4.

east of a16r0 is west of a17r0.

west of a4r0 is east of a17r0.

east of a7r4 is west of a12r4.

southeast of a4r0 is northwest of a18r0.

east of a19r5 is west of a20r5.

southeast of a21r1 is northwest of a22r1.

east of a23r6 is west of a24r6.

east of a24r6 is west of a25r6.

east of a26r3 is west of a27r3.

east of a27r3 is west of a28r3.

northeast of a29r3 is southwest of a30r3.

north of a14r3 is south of a27r3.

south of a31r3 is north of a32r3.

north of a15r1 is south of a22r1.

northeast of a30r3 is southwest of a33r3.

northeast of a31r3 is southwest of a34r3.

west of a31r3 is east of a33r3.

west of a19r5 is east of a35r5.

northwest of a1r0 is southeast of a36r0.

north of a36r0 is south of a37r0.

south of a29r3 is north of a38r3.

northwest of a39r6 is southeast of a40r6.

southwest of a41r1 is northeast of a42r1.

southwest of a43r6 is northeast of a44r6.

southeast of a45r5 is northwest of a46r5.

southeast of a47r3 is northwest of a48r3.

southeast of a49r6 is northwest of a50r6.

southwest of a51r5 is northeast of a52r5.

southwest of a53r3 is northeast of a54r3.

southeast of a55r3 is northwest of a56r3.

north of a57r3 is south of a58r3.

northwest of a59r3 is southeast of a60r3.

southeast of a61r3 is northwest of a62r3.

northeast of a63r5 is southwest of a64r5.

northwest of a65r5 is southeast of a66r5.

northeast of a67r3 is southwest of a68r3.

northwest of a61r3 is southeast of a69r3.

northeast of a70r6 is southwest of a71r6.

north of a72r4 is south of a73r4.

northwest of a74r4 is southeast of a75r4.

southwest of a76r0 is northeast of a77r0.

southwest of a78r4 is northeast of a79r4.

south of a80r4 is north of a81r4.

north of a76r0 is south of a82r0.

northeast of a83r4 is southwest of a84r4.

south of a85r3 is north of a86r3.

northeast of a87r0 is southwest of a88r0.

southeast of a89r0 is northwest of a90r0.

southwest of a2r2 is northeast of a91r2.

southeast of a9r0 is northwest of a92r0.

southeast of a11r2 is northwest of a93r2.

west of a34r3 is east of a94r3.

northwest of a21r1 is southeast of a95r1.

southeast of a18r0 is northwest of a96r0.

southwest of a0r0 is northeast of a97r0.

northwest of a23r6 is southeast of a50r6.

northeast of a20r5 is southwest of a52r5.

northeast of a28r3 is southwest of a54r3.

northwest of a33r3 is southeast of a56r3.

southwest of a26r3 is northeast of a68r3.

southeast of a28r3 is northwest of a69r3.

southwest of a23r6 is northeast of a71r6.

southeast of a25r6 is northwest of a40r6.

northeast of a22r1 is southwest of a42r1.

northeast of a25r6 is southwest of a44r6.

northwest of a35r5 is southeast of a46r5.

northwest of a26r3 is southeast of a48r3.

southwest of a6r4 is northeast of a84r4.

west of a38r3 is east of a86r3.

east of a14r3 is west of a29r3.

southwest of a32r3 is northeast of a58r3.

east of a32r3 is west of a60r3.

northwest of a30r3 is southeast of a62r3.

southwest of a35r5 is northeast of a64r5.

southeast of a20r5 is northwest of a66r5.

southeast of a12r4 is northwest of a75r4.

east of a3r2 is west of a10r2.

east of a1r0 is west of a8r0.

northwest of a5r0 is southeast of a77r0.

east of a13r4 is west of a79r4.

northwest of a13r4 is southeast of a81r4.

east of a5r0 is west of a36r0.

west of a37r0 is east of a82r0.

southwest of a16r0 is northeast of a88r0.

northwest of a16r0 is southeast of a90r0.

south of a7r4 is north of a73r4.

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
the player is in a81r4.

the orientation of the player is 6.
