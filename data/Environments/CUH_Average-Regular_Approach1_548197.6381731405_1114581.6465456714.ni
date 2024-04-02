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

[create a10r0]
a10r0 is a area. "An area (10) in r0".
the printed name of the a10r0 is "Room 0".
Understand "Area 10 in Room 0" as a10r0.
the parent of the a10r0 is r0.
the X of the a10r0 is "548866.0".
the Y of the a10r0 is "1116024.0".

[create a11r0]
a11r0 is a area. "An area (11) in r0".
the printed name of the a11r0 is "Room 0".
Understand "Area 11 in Room 0" as a11r0.
the parent of the a11r0 is r0.
the X of the a11r0 is "548823.94".
the Y of the a11r0 is "1116042.5189999999".

[create a12r0]
a12r0 is a area. "An area (12) in r0".
the printed name of the a12r0 is "Room 0".
Understand "Area 12 in Room 0" as a12r0.
the parent of the a12r0 is r0.
the X of the a12r0 is "549453.0".
the Y of the a12r0 is "1116024.0".

[create a13r0]
a13r0 is a area. "An area (13) in r0".
the printed name of the a13r0 is "Room 0".
Understand "Area 13 in Room 0" as a13r0.
the parent of the a13r0 is r0.
the X of the a13r0 is "549495.06".
the Y of the a13r0 is "1116042.5189999999".

[create a14r0]
a14r0 is a area. "An area (14) in r0".
the printed name of the a14r0 is "Room 0".
Understand "Area 14 in Room 0" as a14r0.
the parent of the a14r0 is r0.
the X of the a14r0 is "550282.0".
the Y of the a14r0 is "1116024.0".

[create a15r0]
a15r0 is a area. "An area (15) in r0".
the printed name of the a15r0 is "Room 0".
Understand "Area 15 in Room 0" as a15r0.
the parent of the a15r0 is r0.
the X of the a15r0 is "550239.94".
the Y of the a15r0 is "1116042.5189999999".

[create a39r0]
a39r0 is a area. "An area (39) in r0".
the printed name of the a39r0 is "Room 0".
Understand "Area 39 in Room 0" as a39r0.
the parent of the a39r0 is r0.
the X of the a39r0 is "547810.0".
the Y of the a39r0 is "1116024.0".

[create a40r0]
a40r0 is a area. "An area (40) in r0".
the printed name of the a40r0 is "Room 0".
Understand "Area 40 in Room 0" as a40r0.
the parent of the a40r0 is r0.
the X of the a40r0 is "547837.2059471365".
the Y of the a40r0 is "1116044.0044052864".

[create a43r0]
a43r0 is a area. "An area (43) in r0".
the printed name of the a43r0 is "Room 0".
Understand "Area 43 in Room 0" as a43r0.
the parent of the a43r0 is r0.
the X of the a43r0 is "550273.0".
the Y of the a43r0 is "1116114.0".

[create a44r0]
a44r0 is a area. "An area (44) in r0".
the printed name of the a44r0 is "Room 0".
Understand "Area 44 in Room 0" as a44r0.
the parent of the a44r0 is r0.
the X of the a44r0 is "550241.9123931625".
the Y of the a44r0 is "1116088.1662393163".

[create a45r0]
a45r0 is a area. "An area (45) in r0".
the printed name of the a45r0 is "Room 0".
Understand "Area 45 in Room 0" as a45r0.
the parent of the a45r0 is r0.
the X of the a45r0 is "548857.0".
the Y of the a45r0 is "1116114.0".

[create a46r0]
a46r0 is a area. "An area (46) in r0".
the printed name of the a46r0 is "Room 0".
Understand "Area 46 in Room 0" as a46r0.
the parent of the a46r0 is r0.
the X of the a46r0 is "548825.9123931625".
the Y of the a46r0 is "1116088.1662393163".

[create a47r0]
a47r0 is a area. "An area (47) in r0".
the printed name of the a47r0 is "Room 0".
Understand "Area 47 in Room 0" as a47r0.
the parent of the a47r0 is r0.
the X of the a47r0 is "549462.0".
the Y of the a47r0 is "1116114.0".

[create a48r0]
a48r0 is a area. "An area (48) in r0".
the printed name of the a48r0 is "Room 0".
Understand "Area 48 in Room 0" as a48r0.
the parent of the a48r0 is r0.
the X of the a48r0 is "549493.0876068376".
the Y of the a48r0 is "1116088.1662393163".

[create a49r0]
a49r0 is a area. "An area (49) in r0".
the printed name of the a49r0 is "Room 0".
Understand "Area 49 in Room 0" as a49r0.
the parent of the a49r0 is r0.
the X of the a49r0 is "547819.0".
the Y of the a49r0 is "1116114.0".

[create a50r0]
a50r0 is a area. "An area (50) in r0".
the printed name of the a50r0 is "Room 0".
Understand "Area 50 in Room 0" as a50r0.
the parent of the a50r0 is r0.
the X of the a50r0 is "547850.0876068377".
the Y of the a50r0 is "1116088.1662393163".

[create a51r0]
a51r0 is a area. "An area (51) in r0".
the printed name of the a51r0 is "Room 0".
Understand "Area 51 in Room 0" as a51r0.
the parent of the a51r0 is r0.
the X of the a51r0 is "548585.1111111111".
the Y of the a51r0 is "1116104.0".

[create a52r0]
a52r0 is a area. "An area (52) in r0".
the printed name of the a52r0 is "Room 0".
Understand "Area 52 in Room 0" as a52r0.
the parent of the a52r0 is r0.
the X of the a52r0 is "548552.1923615087".
the Y of the a52r0 is "1116105.4040895503".

[create a53r0]
a53r0 is a area. "An area (53) in r0".
the printed name of the a53r0 is "Room 0".
Understand "Area 53 in Room 0" as a53r0.
the parent of the a53r0 is r0.
the X of the a53r0 is "548046.0".
the Y of the a53r0 is "1116114.0".

[create a54r0]
a54r0 is a area. "An area (54) in r0".
the printed name of the a54r0 is "Room 0".
Understand "Area 54 in Room 0" as a54r0.
the parent of the a54r0 is r0.
the X of the a54r0 is "548077.826460481".
the Y of the a54r0 is "1116105.767353952".

[create a60r0]
a60r0 is a area. "An area (60) in r0".
the printed name of the a60r0 is "Room 0".
Understand "Area 60 in Room 0" as a60r0.
the parent of the a60r0 is r0.
the X of the a60r0 is "550224.8606168444".
the Y of the a60r0 is "1116059.860290629".

[create a61r0]
a61r0 is a area. "An area (61) in r0".
the printed name of the a61r0 is "Room 0".
Understand "Area 61 in Room 0" as a61r0.
the parent of the a61r0 is r0.
the X of the a61r0 is "548808.8606168446".
the Y of the a61r0 is "1116059.8602906289".

[create a62r0]
a62r0 is a area. "An area (62) in r0".
the printed name of the a62r0 is "Room 0".
Understand "Area 62 in Room 0" as a62r0.
the parent of the a62r0 is r0.
the X of the a62r0 is "549510.1393831554".
the Y of the a62r0 is "1116059.8602906289".

[create a63r0]
a63r0 is a area. "An area (63) in r0".
the printed name of the a63r0 is "Room 0".
Understand "Area 63 in Room 0" as a63r0.
the parent of the a63r0 is r0.
the X of the a63r0 is "547864.5080993522".
the Y of the a63r0 is "1116064.2282217424".

[create a64r0]
a64r0 is a area. "An area (64) in r0".
the printed name of the a64r0 is "Room 0".
Understand "Area 64 in Room 0" as a64r0.
the parent of the a64r0 is r0.
the X of the a64r0 is "548542.7409423069".
the Y of the a64r0 is "1116012.6226577207".

[create a65r0]
a65r0 is a area. "An area (65) in r0".
the printed name of the a65r0 is "Room 0".
Understand "Area 65 in Room 0" as a65r0.
the parent of the a65r0 is r0.
the X of the a65r0 is "548084.4825390012".
the Y of the a65r0 is "1116040.5377844567".

[create a68r0]
a68r0 is a area. "An area (68) in r0".
the printed name of the a68r0 is "Room 0".
Understand "Area 68 in Room 0" as a68r0.
the parent of the a68r0 is r0.
the X of the a68r0 is "548357.5456626685".
the Y of the a68r0 is "1115212.1827022112".

[create a69r0]
a69r0 is a area. "An area (69) in r0".
the printed name of the a69r0 is "Room 0".
Understand "Area 69 in Room 0" as a69r0.
the parent of the a69r0 is r0.
the X of the a69r0 is "548442.1176405045".
the Y of the a69r0 is "1115094.0".

[create a87r0]
a87r0 is a area. "An area (87) in r0".
the printed name of the a87r0 is "Room 0".
Understand "Area 87 in Room 0" as a87r0.
the parent of the a87r0 is r0.
the X of the a87r0 is "549854.46433213".
the Y of the a87r0 is "1115094.0".

[create a88r0]
a88r0 is a area. "An area (88) in r0".
the printed name of the a88r0 is "Room 0".
Understand "Area 88 in Room 0" as a88r0.
the parent of the a88r0 is r0.
the X of the a88r0 is "549867.5".
the Y of the a88r0 is "1115935.7211084906".

[create a89r0]
a89r0 is a area. "An area (89) in r0".
the printed name of the a89r0 is "Room 0".
Understand "Area 89 in Room 0" as a89r0.
the parent of the a89r0 is r0.
the X of the a89r0 is "548354.6853494914".
the Y of the a89r0 is "1115855.1379571522".

[create a0r1]
a0r1 is a area. "An area (0) in r1".
the printed name of the a0r1 is "Room 1".
Understand "Area 0 in Room 1" as a0r1.
the parent of the a0r1 is r1.
the X of the a0r1 is "547052.0".
the Y of the a0r1 is "1114516.0".

[create a1r1]
a1r1 is a area. "An area (1) in r1".
the printed name of the a1r1 is "Room 1".
Understand "Area 1 in Room 1" as a1r1.
the parent of the a1r1 is r1.
the X of the a1r1 is "547018.3349056604".
the Y of the a1r1 is "1114496.278301887".

[create a2r1]
a2r1 is a area. "An area (2) in r1".
the printed name of the a2r1 is "Room 1".
Understand "Area 2 in Room 1" as a2r1.
the parent of the a2r1 is r1.
the X of the a2r1 is "546714.7250283207".
the Y of the a2r1 is "1114226.387822991".

[create a3r1]
a3r1 is a area. "An area (3) in r1".
the printed name of the a3r1 is "Room 1".
Understand "Area 3 in Room 1" as a3r1.
the parent of the a3r1 is r1.
the X of the a3r1 is "546729.4413894466".
the Y of the a3r1 is "1114311.7012868803".

[create a4r1]
a4r1 is a area. "An area (4) in r1".
the printed name of the a4r1 is "Room 1".
Understand "Area 4 in Room 1" as a4r1.
the parent of the a4r1 is r1.
the X of the a4r1 is "545147.0".
the Y of the a4r1 is "1114513.0".

[create a5r1]
a5r1 is a area. "An area (5) in r1".
the printed name of the a5r1 is "Room 1".
Understand "Area 5 in Room 1" as a5r1.
the parent of the a5r1 is r1.
the X of the a5r1 is "545232.1526016355".
the Y of the a5r1 is "1114550.5769230768".

[create a16r1]
a16r1 is a area. "An area (16) in r1".
the printed name of the a16r1 is "Room 1".
Understand "Area 16 in Room 1" as a16r1.
the parent of the a16r1 is r1.
the X of the a16r1 is "549431.0".
the Y of the a16r1 is "1114080.0".

[create a17r1]
a17r1 is a area. "An area (17) in r1".
the printed name of the a17r1 is "Room 1".
Understand "Area 17 in Room 1" as a17r1.
the parent of the a17r1 is r1.
the X of the a17r1 is "549462.0059757691".
the Y of the a17r1 is "1114060.6579076622".

[create a18r1]
a18r1 is a area. "An area (18) in r1".
the printed name of the a18r1 is "Room 1".
Understand "Area 18 in Room 1" as a18r1.
the parent of the a18r1 is r1.
the X of the a18r1 is "548866.0".
the Y of the a18r1 is "1113991.0".

[create a19r1]
a19r1 is a area. "An area (19) in r1".
the printed name of the a19r1 is "Room 1".
Understand "Area 19 in Room 1" as a19r1.
the parent of the a19r1 is r1.
the X of the a19r1 is "548834.7862838916".
the Y of the a19r1 is "1114016.633971292".

[create a20r1]
a20r1 is a area. "An area (20) in r1".
the printed name of the a20r1 is "Room 1".
Understand "Area 20 in Room 1" as a20r1.
the parent of the a20r1 is r1.
the X of the a20r1 is "550282.0".
the Y of the a20r1 is "1113991.0".

[create a21r1]
a21r1 is a area. "An area (21) in r1".
the printed name of the a21r1 is "Room 1".
Understand "Area 21 in Room 1" as a21r1.
the parent of the a21r1 is r1.
the X of the a21r1 is "550250.9579422104".
the Y of the a21r1 is "1114016.6166125855".

[create a22r1]
a22r1 is a area. "An area (22) in r1".
the printed name of the a22r1 is "Room 1".
Understand "Area 22 in Room 1" as a22r1.
the parent of the a22r1 is r1.
the X of the a22r1 is "550291.0".
the Y of the a22r1 is "1114080.0".

[create a23r1]
a23r1 is a area. "An area (23) in r1".
the printed name of the a23r1 is "Room 1".
Understand "Area 23 in Room 1" as a23r1.
the parent of the a23r1 is r1.
the X of the a23r1 is "550248.1485252037".
the Y of the a23r1 is "1114061.8557671143".

[create a24r1]
a24r1 is a area. "An area (24) in r1".
the printed name of the a24r1 is "Room 1".
Understand "Area 24 in Room 1" as a24r1.
the parent of the a24r1 is r1.
the X of the a24r1 is "548875.0".
the Y of the a24r1 is "1114080.0".

[create a25r1]
a25r1 is a area. "An area (25) in r1".
the printed name of the a25r1 is "Room 1".
Understand "Area 25 in Room 1" as a25r1.
the parent of the a25r1 is r1.
the X of the a25r1 is "548832.1485252036".
the Y of the a25r1 is "1114061.8557671143".

[create a26r1]
a26r1 is a area. "An area (26) in r1".
the printed name of the a26r1 is "Room 1".
Understand "Area 26 in Room 1" as a26r1.
the parent of the a26r1 is r1.
the X of the a26r1 is "547899.0".
the Y of the a26r1 is "1113963.0".

[create a27r1]
a27r1 is a area. "An area (27) in r1".
the printed name of the a27r1 is "Room 1".
Understand "Area 27 in Room 1" as a27r1.
the parent of the a27r1 is r1.
the X of the a27r1 is "547863.4166666667".
the Y of the a27r1 is "1114045.8310397551".

[create a28r1]
a28r1 is a area. "An area (28) in r1".
the printed name of the a28r1 is "Room 1".
Understand "Area 28 in Room 1" as a28r1.
the parent of the a28r1 is r1.
the X of the a28r1 is "545737.0".
the Y of the a28r1 is "1113963.0".

[create a29r1]
a29r1 is a area. "An area (29) in r1".
the printed name of the a29r1 is "Room 1".
Understand "Area 29 in Room 1" as a29r1.
the parent of the a29r1 is r1.
the X of the a29r1 is "545773.0".
the Y of the a29r1 is "1114055.3867768594".

[create a30r1]
a30r1 is a area. "An area (30) in r1".
the printed name of the a30r1 is "Room 1".
Understand "Area 30 in Room 1" as a30r1.
the parent of the a30r1 is r1.
the X of the a30r1 is "548015.0".
the Y of the a30r1 is "1114080.0".

[create a31r1]
a31r1 is a area. "An area (31) in r1".
the printed name of the a31r1 is "Room 1".
Understand "Area 31 in Room 1" as a31r1.
the parent of the a31r1 is r1.
the X of the a31r1 is "547971.1165254238".
the Y of the a31r1 is "1114147.7669491526".

[create a32r1]
a32r1 is a area. "An area (32) in r1".
the printed name of the a32r1 is "Room 1".
Understand "Area 32 in Room 1" as a32r1.
the parent of the a32r1 is r1.
the X of the a32r1 is "548024.0".
the Y of the a32r1 is "1113991.0".

[create a33r1]
a33r1 is a area. "An area (33) in r1".
the printed name of the a33r1 is "Room 1".
Understand "Area 33 in Room 1" as a33r1.
the parent of the a33r1 is r1.
the X of the a33r1 is "548055.2137161085".
the Y of the a33r1 is "1114016.633971292".

[create a34r1]
a34r1 is a area. "An area (34) in r1".
the printed name of the a34r1 is "Room 1".
Understand "Area 34 in Room 1" as a34r1.
the parent of the a34r1 is r1.
the X of the a34r1 is "549440.0".
the Y of the a34r1 is "1113991.0".

[create a35r1]
a35r1 is a area. "An area (35) in r1".
the printed name of the a35r1 is "Room 1".
Understand "Area 35 in Room 1" as a35r1.
the parent of the a35r1 is r1.
the X of the a35r1 is "549471.2137161084".
the Y of the a35r1 is "1114016.633971292".

[create a36r1]
a36r1 is a area. "An area (36) in r1".
the printed name of the a36r1 is "Room 1".
Understand "Area 36 in Room 1" as a36r1.
the parent of the a36r1 is r1.
the X of the a36r1 is "548046.0059757694".
the Y of the a36r1 is "1114060.6579076622".

[create a55r1]
a55r1 is a area. "An area (55) in r1".
the printed name of the a55r1 is "Room 1".
Understand "Area 55 in Room 1" as a55r1.
the parent of the a55r1 is r1.
the X of the a55r1 is "546969.0".
the Y of the a55r1 is "1114553.0".

[create a56r1]
a56r1 is a area. "An area (56) in r1".
the printed name of the a56r1 is "Room 1".
Understand "Area 56 in Room 1" as a56r1.
the parent of the a56r1 is r1.
the X of the a56r1 is "546964.972672862".
the Y of the a56r1 is "1114521.7898094188".

[create a67r1]
a67r1 is a area. "An area (67) in r1".
the printed name of the a67r1 is "Room 1".
Understand "Area 67 in Room 1" as a67r1.
the parent of the a67r1 is r1.
the X of the a67r1 is "545344.3872800174".
the Y of the a67r1 is "1114598.6766036483".

[create a71r1]
a71r1 is a area. "An area (71) in r1".
the printed name of the a71r1 is "Room 1".
Understand "Area 71 in Room 1" as a71r1.
the parent of the a71r1 is r1.
the X of the a71r1 is "548817.7817693062".
the Y of the a71r1 is "1114044.9748289345".

[create a72r1]
a72r1 is a area. "An area (72) in r1".
the printed name of the a72r1 is "Room 1".
Understand "Area 72 in Room 1" as a72r1.
the parent of the a72r1 is r1.
the X of the a72r1 is "547846.0218346368".
the Y of the a72r1 is "1114084.5039537789".

[create a73r1]
a73r1 is a area. "An area (73) in r1".
the printed name of the a73r1 is "Room 1".
Understand "Area 73 in Room 1" as a73r1.
the parent of the a73r1 is r1.
the X of the a73r1 is "545790.3005528576".
the Y of the a73r1 is "1114098.8583478578".

[create a74r1]
a74r1 is a area. "An area (74) in r1".
the printed name of the a74r1 is "Room 1".
Understand "Area 74 in Room 1" as a74r1.
the parent of the a74r1 is r1.
the X of the a74r1 is "547954.1282051282".
the Y of the a74r1 is "1114163.3525641025".

[create a75r1]
a75r1 is a area. "An area (75) in r1".
the printed name of the a75r1 is "Room 1".
Understand "Area 75 in Room 1" as a75r1.
the parent of the a75r1 is r1.
the X of the a75r1 is "546997.78205973".
the Y of the a75r1 is "1114489.4273532433".

[create a76r1]
a76r1 is a area. "An area (76) in r1".
the printed name of the a76r1 is "Room 1".
Understand "Area 76 in Room 1" as a76r1.
the parent of the a76r1 is r1.
the X of the a76r1 is "546828.7913371649".
the Y of the a76r1 is "1114363.6185589004".

[create a77r1]
a77r1 is a area. "An area (77) in r1".
the printed name of the a77r1 is "Room 1".
Understand "Area 77 in Room 1" as a77r1.
the parent of the a77r1 is r1.
the X of the a77r1 is "546832.8047738732".
the Y of the a77r1 is "1114360.0669031793".

[create a78r1]
a78r1 is a area. "An area (78) in r1".
the printed name of the a78r1 is "Room 1".
Understand "Area 78 in Room 1" as a78r1.
the parent of the a78r1 is r1.
the X of the a78r1 is "548177.3853736457".
the Y of the a78r1 is "1114447.6808904756".

[create a79r1]
a79r1 is a area. "An area (79) in r1".
the printed name of the a79r1 is "Room 1".
Understand "Area 79 in Room 1" as a79r1.
the parent of the a79r1 is r1.
the X of the a79r1 is "548190.5278947368".
the Y of the a79r1 is "1114509.7793026315".

[create a80r1]
a80r1 is a area. "An area (80) in r1".
the printed name of the a80r1 is "Room 1".
Understand "Area 80 in Room 1" as a80r1.
the parent of the a80r1 is r1.
the X of the a80r1 is "548069.389199255".
the Y of the a80r1 is "1114040.2597765361".

[create a81r1]
a81r1 is a area. "An area (81) in r1".
the printed name of the a81r1 is "Room 1".
Understand "Area 81 in Room 1" as a81r1.
the parent of the a81r1 is r1.
the X of the a81r1 is "549485.389199255".
the Y of the a81r1 is "1114040.2597765361".

[create a82r1]
a82r1 is a area. "An area (82) in r1".
the printed name of the a82r1 is "Room 1".
Understand "Area 82 in Room 1" as a82r1.
the parent of the a82r1 is r1.
the X of the a82r1 is "550233.8279210806".
the Y of the a82r1 is "1114045.02905727".

[create a83r1]
a83r1 is a area. "An area (83) in r1".
the printed name of the a83r1 is "Room 1".
Understand "Area 83 in Room 1" as a83r1.
the parent of the a83r1 is r1.
the X of the a83r1 is "548204.689081676".
the Y of the a83r1 is "1114653.5152729584".

[create a84r1]
a84r1 is a area. "An area (84) in r1".
the printed name of the a84r1 is "Room 1".
Understand "Area 84 in Room 1" as a84r1.
the parent of the a84r1 is r1.
the X of the a84r1 is "548450.2380697202".
the Y of the a84r1 is "1114631.0".

[create a85r1]
a85r1 is a area. "An area (85) in r1".
the printed name of the a85r1 is "Room 1".
Understand "Area 85 in Room 1" as a85r1.
the parent of the a85r1 is r1.
the X of the a85r1 is "548450.4968411552".
the Y of the a85r1 is "1114699.0".

[create a86r1]
a86r1 is a area. "An area (86) in r1".
the printed name of the a86r1 is "Room 1".
Understand "Area 86 in Room 1" as a86r1.
the parent of the a86r1 is r1.
the X of the a86r1 is "548450.7908675541".
the Y of the a86r1 is "1114767.0".

[create a91r1]
a91r1 is a area. "An area (91) in r1".
the printed name of the a91r1 is "Room 1".
Understand "Area 91 in Room 1" as a91r1.
the parent of the a91r1 is r1.
the X of the a91r1 is "548449.6732851986".
the Y of the a91r1 is "1114427.0".

[create a92r1]
a92r1 is a area. "An area (92) in r1".
the printed name of the a92r1 is "Room 1".
Understand "Area 92 in Room 1" as a92r1.
the parent of the a92r1 is r1.
the X of the a92r1 is "548449.8262917418".
the Y of the a92r1 is "1114495.0".

[create a93r1]
a93r1 is a area. "An area (93) in r1".
the printed name of the a93r1 is "Room 1".
Understand "Area 93 in Room 1" as a93r1.
the parent of the a93r1 is r1.
the X of the a93r1 is "548197.6381731405".
the Y of the a93r1 is "1114581.6465456714".

[create a94r1]
a94r1 is a area. "An area (94) in r1".
the printed name of the a94r1 is "Room 1".
Understand "Area 94 in Room 1" as a94r1.
the parent of the a94r1 is r1.
the X of the a94r1 is "548450.0145532491".
the Y of the a94r1 is "1114563.0".

[create a95r1]
a95r1 is a area. "An area (95) in r1".
the printed name of the a95r1 is "Room 1".
Understand "Area 95 in Room 1" as a95r1.
the parent of the a95r1 is r1.
the X of the a95r1 is "546874.9776676891".
the Y of the a95r1 is "1114234.4869705806".

[create a96r1]
a96r1 is a area. "An area (96) in r1".
the printed name of the a96r1 is "Room 1".
Understand "Area 96 in Room 1" as a96r1.
the parent of the a96r1 is r1.
the X of the a96r1 is "549867.5".
the Y of the a96r1 is "1114168.1939858492".

[create a97r1]
a97r1 is a area. "An area (97) in r1".
the printed name of the a97r1 is "Room 1".
Understand "Area 97 in Room 1" as a97r1.
the parent of the a97r1 is r1.
the X of the a97r1 is "548451.0".
the Y of the a97r1 is "1114168.5778301884".

[create a6r2]
a6r2 is a area. "An area (6) in r2".
the printed name of the a6r2 is "Room 2".
Understand "Area 6 in Room 2" as a6r2.
the parent of the a6r2 is r2.
the X of the a6r2 is "545147.0".
the Y of the a6r2 is "1115490.0".

[create a7r2]
a7r2 is a area. "An area (7) in r2".
the printed name of the a7r2 is "Room 2".
Understand "Area 7 in Room 2" as a7r2.
the parent of the a7r2 is r2.
the X of the a7r2 is "545246.9343447648".
the Y of the a7r2 is "1115452.423076923".

[create a8r2]
a8r2 is a area. "An area (8) in r2".
the printed name of the a8r2 is "Room 2".
Understand "Area 8 in Room 2" as a8r2.
the parent of the a8r2 is r2.
the X of the a8r2 is "547202.0".
the Y of the a8r2 is "1116024.0".

[create a9r2]
a9r2 is a area. "An area (9) in r2".
the printed name of the a9r2 is "Room 2".
Understand "Area 9 in Room 2" as a9r2.
the parent of the a9r2 is r2.
the X of the a9r2 is "547159.9400000001".
the Y of the a9r2 is "1116042.5189999999".

[create a37r2]
a37r2 is a area. "An area (37) in r2".
the printed name of the a37r2 is "Room 2".
Understand "Area 37 in Room 2" as a37r2.
the parent of the a37r2 is r2.
the X of the a37r2 is "545731.0".
the Y of the a37r2 is "1116132.0".

[create a38r2]
a38r2 is a area. "An area (38) in r2".
the printed name of the a38r2 is "Room 2".
Understand "Area 38 in Room 2" as a38r2.
the parent of the a38r2 is r2.
the X of the a38r2 is "545767.2142857146".
the Y of the a38r2 is "1116045.7283246347".

[create a41r2]
a41r2 is a area. "An area (41) in r2".
the printed name of the a41r2 is "Room 2".
Understand "Area 41 in Room 2" as a41r2.
the parent of the a41r2 is r2.
the X of the a41r2 is "547193.0".
the Y of the a41r2 is "1116114.0".

[create a42r2]
a42r2 is a area. "An area (42) in r2".
the printed name of the a42r2 is "Room 2".
Understand "Area 42 in Room 2" as a42r2.
the parent of the a42r2 is r2.
the X of the a42r2 is "547161.9123931625".
the Y of the a42r2 is "1116088.1662393163".

[create a57r2]
a57r2 is a area. "An area (57) in r2".
the printed name of the a57r2 is "Room 2".
Understand "Area 57 in Room 2" as a57r2.
the parent of the a57r2 is r2.
the X of the a57r2 is "546969.0".
the Y of the a57r2 is "1115625.0".

[create a58r2]
a58r2 is a area. "An area (58) in r2".
the printed name of the a58r2 is "Room 2".
Understand "Area 58 in Room 2" as a58r2.
the parent of the a58r2 is r2.
the X of the a58r2 is "546930.24644506".
the Y of the a58r2 is "1115648.7415304407".

[create a59r2]
a59r2 is a area. "An area (59) in r2".
the printed name of the a59r2 is "Room 2".
Understand "Area 59 in Room 2" as a59r2.
the parent of the a59r2 is r2.
the X of the a59r2 is "547144.8606168445".
the Y of the a59r2 is "1116059.860290629".

[create a66r2]
a66r2 is a area. "An area (66) in r2".
the printed name of the a66r2 is "Room 2".
Understand "Area 66 in Room 2" as a66r2.
the parent of the a66r2 is r2.
the X of the a66r2 is "545781.8275346495".
the Y of the a66r2 is "1116014.3004094234".

[create a70r2]
a70r2 is a area. "An area (70) in r2".
the printed name of the a70r2 is "Room 2".
Understand "Area 70 in Room 2" as a70r2.
the parent of the a70r2 is r2.
the X of the a70r2 is "545323.5320229848".
the Y of the a70r2 is "1115421.9653398965".

[create a90r2]
a90r2 is a area. "An area (90) in r2".
the printed name of the a90r2 is "Room 2".
Understand "Area 90 in Room 2" as a90r2.
the parent of the a90r2 is r2.
the X of the a90r2 is "546791.335305074".
the Y of the a90r2 is "1115700.6044317854".

[create a98r2]
a98r2 is a area. "An area (98) in r2".
the printed name of the a98r2 is "Room 2".
Understand "Area 98 in Room 2" as a98r2.
the parent of the a98r2 is r2.
the X of the a98r2 is "546790.0".
the Y of the a98r2 is "1115937.8051233583".

[create d0]
d0 is a door. "A door between a68r0 and a83r1".
d0 is south of a68r0 and north of a83r1.

[create d1]
d1 is a door. "A door between a69r0 and a86r1".
d1 is south of a69r0 and north of a86r1.

[create d2]
d2 is a door. "A door between a67r1 and a70r2".
d2 is north of a67r1 and south of a70r2.

[create d3]
d3 is a door. "A door between a76r1 and a90r2".
d3 is north of a76r1 and south of a90r2.

[create d4]
d4 is a door. "A door between a87r0 and a96r1".
d4 is south of a87r0 and north of a96r1.

southwest of a0r1 is northeast of a1r1.

north of a2r1 is south of a3r1.

northeast of a4r1 is southwest of a5r1.

east of a6r2 is west of a7r2.

northwest of a8r2 is southeast of a9r2.

northwest of a10r0 is southeast of a11r0.

northeast of a12r0 is southwest of a13r0.

northwest of a14r0 is southeast of a15r0.

southeast of a16r1 is northwest of a17r1.

northwest of a18r1 is southeast of a19r1.

northwest of a20r1 is southeast of a21r1.

southwest of a22r1 is northeast of a23r1.

southwest of a24r1 is northeast of a25r1.

northwest of a26r1 is southeast of a27r1.

north of a28r1 is south of a29r1.

northwest of a30r1 is southeast of a31r1.

northeast of a32r1 is southwest of a33r1.

northeast of a34r1 is southwest of a35r1.

southeast of a30r1 is northwest of a36r1.

southeast of a37r2 is northwest of a38r2.

northeast of a39r0 is southwest of a40r0.

southwest of a41r2 is northeast of a42r2.

southwest of a43r0 is northeast of a44r0.

southwest of a45r0 is northeast of a46r0.

southeast of a47r0 is northwest of a48r0.

southeast of a49r0 is northwest of a50r0.

west of a51r0 is east of a52r0.

east of a53r0 is west of a54r0.

south of a55r1 is north of a56r1.

northwest of a57r2 is southeast of a58r2.

southwest of a42r2 is northeast of a59r2.

southwest of a44r0 is northeast of a60r0.

southwest of a46r0 is northeast of a61r0.

southeast of a48r0 is northwest of a62r0.

southeast of a50r0 is northwest of a63r0.

south of a52r0 is north of a64r0.

south of a54r0 is north of a65r0.

southeast of a38r2 is northwest of a66r2.

northeast of a5r1 is southwest of a67r1.

southeast of a68r0 is northwest of a69r0.

east of a7r2 is west of a70r2.

northwest of a9r2 is southeast of a59r2.

northwest of a11r0 is southeast of a61r0.

northeast of a13r0 is southwest of a62r0.

northwest of a15r0 is southeast of a60r0.

northeast of a40r0 is southwest of a63r0.

southwest of a25r1 is northeast of a71r1.

northwest of a27r1 is southeast of a72r1.

north of a29r1 is south of a73r1.

northwest of a31r1 is southeast of a74r1.

west of a1r1 is east of a75r1.

northeast of a3r1 is southwest of a76r1.

southeast of a76r1 is northwest of a77r1.

north of a78r1 is south of a79r1.

northeast of a33r1 is southwest of a80r1.

northeast of a35r1 is southwest of a81r1.

southeast of a36r1 is northwest of a80r1.

southeast of a17r1 is northwest of a81r1.

northwest of a19r1 is southeast of a71r1.

northwest of a21r1 is southeast of a82r1.

southwest of a23r1 is northeast of a82r1.

north of a84r1 is south of a85r1.

north of a85r1 is south of a86r1.

east of a69r0 is west of a87r0.

north of a87r0 is south of a88r0.

north of a68r0 is south of a89r0.

southwest of a66r2 is northeast of a70r2.

southwest of a75r1 is northeast of a77r1.

southeast of a56r1 is northwest of a75r1.

north of a91r1 is south of a92r1.

north of a79r1 is south of a93r1.

north of a92r1 is south of a94r1.

south of a83r1 is north of a93r1.

south of a84r1 is north of a94r1.

east of a73r1 is west of a95r1.

west of a72r1 is east of a95r1.

northeast of a72r1 is southwest of a74r1.

southeast of a67r1 is northwest of a73r1.

northeast of a74r1 is southwest of a78r1.

south of a91r1 is north of a97r1.

south of a77r1 is north of a95r1.

west of a82r1 is east of a96r1.

west of a71r1 is east of a97r1.

east of a80r1 is west of a97r1.

east of a81r1 is west of a96r1.

east of a66r2 is west of a98r2.

west of a59r2 is east of a98r2.

west of a61r0 is east of a64r0.

west of a58r2 is east of a90r2.

north of a90r2 is south of a98r2.

east of a62r0 is west of a88r0.

west of a60r0 is east of a88r0.

east of a63r0 is west of a65r0.

southwest of a64r0 is northeast of a89r0.

southeast of a65r0 is northwest of a89r0.

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
the player is in a93r1.

the orientation of the player is 6.
