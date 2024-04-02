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
the X of the a0r0 is "548907.1865386872".
the Y of the a0r0 is "1435259.6370670425".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "548926.7466061565".
the Y of the a1r0 is "1435219.0337490134".

[create a8r0]
a8r0 is a area. "An area (8) in r0".
the printed name of the a8r0 is "Room 0".
Understand "Area 8 in Room 0" as a8r0.
the parent of the a8r0 is r0.
the X of the a8r0 is "548235.5080993524".
the Y of the a8r0 is "1435264.2282217427".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "548589.8704484986".
the Y of the a9r0 is "1435176.4146870645".

[create a13r0]
a13r0 is a area. "An area (13) in r0".
the printed name of the a13r0 is "Room 0".
Understand "Area 13 in Room 0" as a13r0.
the parent of the a13r0 is r0.
the X of the a13r0 is "548620.0088954244".
the Y of the a13r0 is "1434579.6581090507".

[create a14r0]
a14r0 is a area. "An area (14) in r0".
the printed name of the a14r0 is "Room 0".
Understand "Area 14 in Room 0" as a14r0.
the parent of the a14r0 is r0.
the X of the a14r0 is "548682.57233583".
the Y of the a14r0 is "1434790.1743150589".

[create a16r0]
a16r0 is a area. "An area (16) in r0".
the printed name of the a16r0 is "Room 0".
Understand "Area 16 in Room 0" as a16r0.
the parent of the a16r0 is r0.
the X of the a16r0 is "548821.2791834099".
the Y of the a16r0 is "1434935.152862847".

[create a17r0]
a17r0 is a area. "An area (17) in r0".
the printed name of the a17r0 is "Room 0".
Understand "Area 17 in Room 0" as a17r0.
the parent of the a17r0 is r0.
the X of the a17r0 is "549027.6906284664".
the Y of the a17r0 is "1435138.7025811374".

[create a19r0]
a19r0 is a area. "An area (19) in r0".
the printed name of the a19r0 is "Room 0".
Understand "Area 19 in Room 0" as a19r0.
the parent of the a19r0 is r0.
the X of the a19r0 is "550172.4801166632".
the Y of the a19r0 is "1435211.8184044985".

[create a49r0]
a49r0 is a area. "An area (49) in r0".
the printed name of the a49r0 is "Room 0".
Understand "Area 49 in Room 0" as a49r0.
the parent of the a49r0 is r0.
the X of the a49r0 is "548181.0".
the Y of the a49r0 is "1435224.0".

[create a50r0]
a50r0 is a area. "An area (50) in r0".
the printed name of the a50r0 is "Room 0".
Understand "Area 50 in Room 0" as a50r0.
the parent of the a50r0 is r0.
the X of the a50r0 is "548208.2059471366".
the Y of the a50r0 is "1435244.0044052864".

[create a53r0]
a53r0 is a area. "An area (53) in r0".
the printed name of the a53r0 is "Room 0".
Understand "Area 53 in Room 0" as a53r0.
the parent of the a53r0 is r0.
the X of the a53r0 is "548955.0".
the Y of the a53r0 is "1435314.0".

[create a54r0]
a54r0 is a area. "An area (54) in r0".
the printed name of the a54r0 is "Room 0".
Understand "Area 54 in Room 0" as a54r0.
the parent of the a54r0 is r0.
the X of the a54r0 is "548924.1050372066".
the Y of the a54r0 is "1435287.7894485786".

[create a59r0]
a59r0 is a area. "An area (59) in r0".
the printed name of the a59r0 is "Room 0".
Understand "Area 59 in Room 0" as a59r0.
the parent of the a59r0 is r0.
the X of the a59r0 is "548190.0".
the Y of the a59r0 is "1435314.0".

[create a60r0]
a60r0 is a area. "An area (60) in r0".
the printed name of the a60r0 is "Room 0".
Understand "Area 60 in Room 0" as a60r0.
the parent of the a60r0 is r0.
the X of the a60r0 is "548221.0876068375".
the Y of the a60r0 is "1435288.1662393163".

[create a63r0]
a63r0 is a area. "An area (63) in r0".
the printed name of the a63r0 is "Room 0".
Understand "Area 63 in Room 0" as a63r0.
the parent of the a63r0 is r0.
the X of the a63r0 is "548495.0".
the Y of the a63r0 is "1434699.5736434108".

[create a64r0]
a64r0 is a area. "An area (64) in r0".
the printed name of the a64r0 is "Room 0".
Understand "Area 64 in Room 0" as a64r0.
the parent of the a64r0 is r0.
the X of the a64r0 is "548563.4720118984".
the Y of the a64r0 is "1434598.5833333335".

[create a67r0]
a67r0 is a area. "An area (67) in r0".
the printed name of the a67r0 is "Room 0".
Understand "Area 67 in Room 0" as a67r0.
the parent of the a67r0 is r0.
the X of the a67r0 is "548707.4869108638".
the Y of the a67r0 is "1434977.947968496".

[create a68r0]
a68r0 is a area. "An area (68) in r0".
the printed name of the a68r0 is "Room 0".
Understand "Area 68 in Room 0" as a68r0.
the parent of the a68r0 is r0.
the X of the a68r0 is "548729.1442388982".
the Y of the a68r0 is "1435005.4807199002".

[create a69r0]
a69r0 is a area. "An area (69) in r0".
the printed name of the a69r0 is "Room 0".
Understand "Area 69 in Room 0" as a69r0.
the parent of the a69r0 is r0.
the X of the a69r0 is "548965.0".
the Y of the a69r0 is "1435224.0".

[create a70r0]
a70r0 is a area. "An area (70) in r0".
the printed name of the a70r0 is "Room 0".
Understand "Area 70 in Room 0" as a70r0.
the parent of the a70r0 is r0.
the X of the a70r0 is "548996.8151310921".
the Y of the a70r0 is "1435207.280319024".

[create a80r0]
a80r0 is a area. "An area (80) in r0".
the printed name of the a80r0 is "Room 0".
Understand "Area 80 in Room 0" as a80r0.
the parent of the a80r0 is r0.
the X of the a80r0 is "550205.8444444444".
the Y of the a80r0 is "1435197.0".

[create a12r1]
a12r1 is a area. "An area (12) in r1".
the printed name of the a12r1 is "Room 1".
Understand "Area 12 in Room 1" as a12r1.
the parent of the a12r1 is r1.
the X of the a12r1 is "550396.652692378".
the Y of the a12r1 is "1433919.6370413492".

[create a20r1]
a20r1 is a area. "An area (20) in r1".
the printed name of the a20r1 is "Room 1".
Understand "Area 20 in Room 1" as a20r1.
the parent of the a20r1 is r1.
the X of the a20r1 is "548646.2082942504".
the Y of the a20r1 is "1433718.0948418116".

[create a21r1]
a21r1 is a area. "An area (21) in r1".
the printed name of the a21r1 is "Room 1".
Understand "Area 21 in Room 1" as a21r1.
the parent of the a21r1 is r1.
the X of the a21r1 is "549000.3894933735".
the Y of the a21r1 is "1433295.2069205644".

[create a23r1]
a23r1 is a area. "An area (23) in r1".
the printed name of the a23r1 is "Room 1".
Understand "Area 23 in Room 1" as a23r1.
the parent of the a23r1 is r1.
the X of the a23r1 is "546869.6571255578".
the Y of the a23r1 is "1433237.8159683137".

[create a24r1]
a24r1 is a area. "An area (24) in r1".
the printed name of the a24r1 is "Room 1".
Understand "Area 24 in Room 1" as a24r1.
the parent of the a24r1 is r1.
the X of the a24r1 is "546921.5107286529".
the Y of the a24r1 is "1433340.449436728".

[create a25r1]
a25r1 is a area. "An area (25) in r1".
the printed name of the a25r1 is "Room 1".
Understand "Area 25 in Room 1" as a25r1.
the parent of the a25r1 is r1.
the X of the a25r1 is "546890.3355178426".
the Y of the a25r1 is "1433421.7819760663".

[create a26r1]
a26r1 is a area. "An area (26) in r1".
the printed name of the a26r1 is "Room 1".
Understand "Area 26 in Room 1" as a26r1.
the parent of the a26r1 is r1.
the X of the a26r1 is "547096.4953543339".
the Y of the a26r1 is "1433659.5377306952".

[create a29r1]
a29r1 is a area. "An area (29) in r1".
the printed name of the a29r1 is "Room 1".
Understand "Area 29 in Room 1" as a29r1.
the parent of the a29r1 is r1.
the X of the a29r1 is "548216.4349012829".
the Y of the a29r1 is "1433284.4525765597".

[create a30r1]
a30r1 is a area. "An area (30) in r1".
the printed name of the a30r1 is "Room 1".
Understand "Area 30 in Room 1" as a30r1.
the parent of the a30r1 is r1.
the X of the a30r1 is "547057.8020768753".
the Y of the a30r1 is "1433226.792467452".

[create a39r1]
a39r1 is a area. "An area (39) in r1".
the printed name of the a39r1 is "Room 1".
Understand "Area 39 in Room 1" as a39r1.
the parent of the a39r1 is r1.
the X of the a39r1 is "548270.0".
the Y of the a39r1 is "1433163.0".

[create a40r1]
a40r1 is a area. "An area (40) in r1".
the printed name of the a40r1 is "Room 1".
Understand "Area 40 in Room 1" as a40r1.
the parent of the a40r1 is r1.
the X of the a40r1 is "548234.4166666666".
the Y of the a40r1 is "1433243.1491127054".

[create a41r1]
a41r1 is a area. "An area (41) in r1".
the printed name of the a41r1 is "Room 1".
Understand "Area 41 in Room 1" as a41r1.
the parent of the a41r1 is r1.
the X of the a41r1 is "548936.0".
the Y of the a41r1 is "1433151.0".

[create a42r1]
a42r1 is a area. "An area (42) in r1".
the printed name of the a42r1 is "Room 1".
Understand "Area 42 in Room 1" as a42r1.
the parent of the a42r1 is r1.
the X of the a42r1 is "548972.0".
the Y of the a42r1 is "1433231.3913903504".

[create a43r1]
a43r1 is a area. "An area (43) in r1".
the printed name of the a43r1 is "Room 1".
Understand "Area 43 in Room 1" as a43r1.
the parent of the a43r1 is r1.
the X of the a43r1 is "550561.0".
the Y of the a43r1 is "1433864.0".

[create a44r1]
a44r1 is a area. "An area (44) in r1".
the printed name of the a44r1 is "Room 1".
Understand "Area 44 in Room 1" as a44r1.
the parent of the a44r1 is r1.
the X of the a44r1 is "550448.6812015504".
the Y of the a44r1 is "1433903.1666666665".

[create a77r1]
a77r1 is a area. "An area (77) in r1".
the printed name of the a77r1 is "Room 1".
Understand "Area 77 in Room 1" as a77r1.
the parent of the a77r1 is r1.
the X of the a77r1 is "547092.0".
the Y of the a77r1 is "1433200.0".

[create a78r1]
a78r1 is a area. "An area (78) in r1".
the printed name of the a78r1 is "Room 1".
Understand "Area 78 in Room 1" as a78r1.
the parent of the a78r1 is r1.
the X of the a78r1 is "546846.0".
the Y of the a78r1 is "1433172.0".

[create a11r2]
a11r2 is a area. "An area (11) in r2".
the printed name of the a11r2 is "Room 2".
Understand "Area 11 in Room 2" as a11r2.
the parent of the a11r2 is r2.
the X of the a11r2 is "550398.5359105705".
the Y of the a11r2 is "1434740.6431929711".

[create a15r2]
a15r2 is a area. "An area (15) in r2".
the printed name of the a15r2 is "Room 2".
Understand "Area 15 in Room 2" as a15r2.
the parent of the a15r2 is r2.
the X of the a15r2 is "550252.3354772709".
the Y of the a15r2 is "1435125.6656008668".

[create a65r2]
a65r2 is a area. "An area (65) in r2".
the printed name of the a65r2 is "Room 2".
Understand "Area 65 in Room 2" as a65r2.
the parent of the a65r2 is r2.
the X of the a65r2 is "550561.0".
the Y of the a65r2 is "1434804.0".

[create a66r2]
a66r2 is a area. "An area (66) in r2".
the printed name of the a66r2 is "Room 2".
Understand "Area 66 in Room 2" as a66r2.
the parent of the a66r2 is r2.
the X of the a66r2 is "550466.4970438057".
the Y of the a66r2 is "1434764.8333333333".

[create a79r2]
a79r2 is a area. "An area (79) in r2".
the printed name of the a79r2 is "Room 2".
Understand "Area 79 in Room 2" as a79r2.
the parent of the a79r2 is r2.
the X of the a79r2 is "550239.0".
the Y of the a79r2 is "1435160.31147541".

[create a6r3]
a6r3 is a area. "An area (6) in r3".
the printed name of the a6r3 is "Room 3".
Understand "Area 6 in Room 3" as a6r3.
the parent of the a6r3 is r3.
the X of the a6r3 is "545168.092671901".
the Y of the a6r3 is "1435259.8065726862".

[create a7r3]
a7r3 is a area. "An area (7) in r3".
the printed name of the a7r3 is "Room 3".
Understand "Area 7 in Room 3" as a7r3.
the parent of the a7r3 is r3.
the X of the a7r3 is "545503.8981420202".
the Y of the a7r3 is "1435156.089074554".

[create a18r3]
a18r3 is a area. "An area (18) in r3".
the printed name of the a18r3 is "Room 3".
Understand "Area 18 in Room 3" as a18r3.
the parent of the a18r3 is r3.
the X of the a18r3 is "545554.0365723592".
the Y of the a18r3 is "1435032.6014533548".

[create a55r3]
a55r3 is a area. "An area (55) in r3".
the printed name of the a55r3 is "Room 3".
Understand "Area 55 in Room 3" as a55r3.
the parent of the a55r3 is r3.
the X of the a55r3 is "545120.0".
the Y of the a55r3 is "1435314.0".

[create a56r3]
a56r3 is a area. "An area (56) in r3".
the printed name of the a56r3 is "Room 3".
Understand "Area 56 in Room 3" as a56r3.
the parent of the a56r3 is r3.
the X of the a56r3 is "545150.9153780071".
the Y of the a56r3 is "1435288.1834621993".

[create a71r3]
a71r3 is a area. "An area (71) in r3".
the printed name of the a71r3 is "Room 3".
Understand "Area 71 in Room 3" as a71r3.
the parent of the a71r3 is r3.
the X of the a71r3 is "545578.0".
the Y of the a71r3 is "1435304.0".

[create a72r3]
a72r3 is a area. "An area (72) in r3".
the printed name of the a72r3 is "Room 3".
Understand "Area 72 in Room 3" as a72r3.
the parent of the a72r3 is r3.
the X of the a72r3 is "545544.8750000001".
the Y of the a72r3 is "1435230.0006598176".

[create a73r3]
a73r3 is a area. "An area (73) in r3".
the printed name of the a73r3 is "Room 3".
Understand "Area 73 in Room 3" as a73r3.
the parent of the a73r3 is r3.
the X of the a73r3 is "545111.0".
the Y of the a73r3 is "1435224.0".

[create a74r3]
a74r3 is a area. "An area (74) in r3".
the printed name of the a74r3 is "Room 3".
Understand "Area 74 in Room 3" as a74r3.
the parent of the a74r3 is r3.
the X of the a74r3 is "545153.0599999999".
the Y of the a74r3 is "1435242.5189999999".

[create a22r4]
a22r4 is a area. "An area (22) in r4".
the printed name of the a22r4 is "Room 4".
Understand "Area 22 in Room 4" as a22r4.
the parent of the a22r4 is r4.
the X of the a22r4 is "545483.4387151967".
the Y of the a22r4 is "1433330.6903873074".

[create a27r4]
a27r4 is a area. "An area (27) in r4".
the printed name of the a27r4 is "Room 4".
Understand "Area 27 in Room 4" as a27r4.
the parent of the a27r4 is r4.
the X of the a27r4 is "545839.7586021505".
the Y of the a27r4 is "1433240.0134408602".

[create a28r4]
a28r4 is a area. "An area (28) in r4".
the printed name of the a28r4 is "Room 4".
Understand "Area 28 in Room 4" as a28r4.
the parent of the a28r4 is r4.
the X of the a28r4 is "545119.3147627445".
the Y of the a28r4 is "1433240.2898139765".

[create a31r4]
a31r4 is a area. "An area (31) in r4".
the printed name of the a31r4 is "Room 4".
Understand "Area 31 in Room 4" as a31r4.
the parent of the a31r4 is r4.
the X of the a31r4 is "545885.0".
the Y of the a31r4 is "1433191.0".

[create a32r4]
a32r4 is a area. "An area (32) in r4".
the printed name of the a32r4 is "Room 4".
Understand "Area 32 in Room 4" as a32r4.
the parent of the a32r4 is r4.
the X of the a32r4 is "545853.7862838916".
the Y of the a32r4 is "1433216.6339712918".

[create a33r4]
a33r4 is a area. "An area (33) in r4".
the printed name of the a33r4 is "Room 4".
Understand "Area 33 in Room 4" as a33r4.
the parent of the a33r4 is r4.
the X of the a33r4 is "545074.0".
the Y of the a33r4 is "1433191.0".

[create a34r4]
a34r4 is a area. "An area (34) in r4".
the printed name of the a34r4 is "Room 4".
Understand "Area 34 in Room 4" as a34r4.
the parent of the a34r4 is r4.
the X of the a34r4 is "545105.0420577895".
the Y of the a34r4 is "1433216.6166125853".

[create a35r4]
a35r4 is a area. "An area (35) in r4".
the printed name of the a35r4 is "Room 4".
Understand "Area 35 in Room 4" as a35r4.
the parent of the a35r4 is r4.
the X of the a35r4 is "545894.0".
the Y of the a35r4 is "1433280.0".

[create a36r4]
a36r4 is a area. "An area (36) in r4".
the printed name of the a36r4 is "Room 4".
Understand "Area 36 in Room 4" as a36r4.
the parent of the a36r4 is r4.
the X of the a36r4 is "545866.849158728".
the Y of the a36r4 is "1433260.2680626004".

[create a37r4]
a37r4 is a area. "An area (37) in r4".
the printed name of the a37r4 is "Room 4".
Understand "Area 37 in Room 4" as a37r4.
the parent of the a37r4 is r4.
the X of the a37r4 is "545065.0".
the Y of the a37r4 is "1433280.0".

[create a38r4]
a38r4 is a area. "An area (38) in r4".
the printed name of the a38r4 is "Room 4".
Understand "Area 38 in Room 4" as a38r4.
the parent of the a38r4 is r4.
the X of the a38r4 is "545096.676963553".
the Y of the a38r4 is "1433260.7257603593".

[create a2r5]
a2r5 is a area. "An area (2) in r5".
the printed name of the a2r5 is "Room 5".
Understand "Area 2 in Room 5" as a2r5.
the parent of the a2r5 is r5.
the X of the a2r5 is "546790.6588446487".
the Y of the a2r5 is "1435245.1990633644".

[create a3r5]
a3r5 is a area. "An area (3) in r5".
the printed name of the a3r5 is "Room 5".
Understand "Area 3 in Room 5" as a3r5.
the parent of the a3r5 is r5.
the X of the a3r5 is "547083.8405675022".
the Y of the a3r5 is "1435071.7652590368".

[create a4r5]
a4r5 is a area. "An area (4) in r5".
the printed name of the a4r5 is "Room 5".
Understand "Area 4 in Room 5" as a4r5.
the parent of the a4r5 is r5.
the X of the a4r5 is "547292.0576299748".
the Y of the a4r5 is "1435227.4593219238".

[create a5r5]
a5r5 is a area. "An area (5) in r5".
the printed name of the a5r5 is "Room 5".
Understand "Area 5 in Room 5" as a5r5.
the parent of the a5r5 is r5.
the X of the a5r5 is "547515.8606168448".
the Y of the a5r5 is "1435259.8602906289".

[create a10r5]
a10r5 is a area. "An area (10) in r5".
the printed name of the a10r5 is "Room 5".
Understand "Area 10 in Room 5" as a10r5.
the parent of the a10r5 is r5.
the X of the a10r5 is "546612.0308738074".
the Y of the a10r5 is "1435263.874721339".

[create a45r5]
a45r5 is a area. "An area (45) in r5".
the printed name of the a45r5 is "Room 5".
Understand "Area 45 in Room 5" as a45r5.
the parent of the a45r5 is r5.
the X of the a45r5 is "546753.0".
the Y of the a45r5 is "1435314.0".

[create a46r5]
a46r5 is a area. "An area (46) in r5".
the printed name of the a46r5 is "Room 5".
Understand "Area 46 in Room 5" as a46r5.
the parent of the a46r5 is r5.
the X of the a46r5 is "546784.8371948844".
the Y of the a46r5 is "1435305.9745279343".

[create a47r5]
a47r5 is a area. "An area (47) in r5".
the printed name of the a47r5 is "Room 5".
Understand "Area 47 in Room 5" as a47r5.
the parent of the a47r5 is r5.
the X of the a47r5 is "546558.0".
the Y of the a47r5 is "1435224.0".

[create a48r5]
a48r5 is a area. "An area (48) in r5".
the printed name of the a48r5 is "Room 5".
Understand "Area 48 in Room 5" as a48r5.
the parent of the a48r5 is r5.
the X of the a48r5 is "546585.2059471365".
the Y of the a48r5 is "1435244.0044052864".

[create a51r5]
a51r5 is a area. "An area (51) in r5".
the printed name of the a51r5 is "Room 5".
Understand "Area 51 in Room 5" as a51r5.
the parent of the a51r5 is r5.
the X of the a51r5 is "546567.0".
the Y of the a51r5 is "1435314.0".

[create a52r5]
a52r5 is a area. "An area (52) in r5".
the printed name of the a52r5 is "Room 5".
Understand "Area 52 in Room 5" as a52r5.
the parent of the a52r5 is r5.
the X of the a52r5 is "546596.5922122437".
the Y of the a52r5 is "1435288.3157787758".

[create a57r5]
a57r5 is a area. "An area (57) in r5".
the printed name of the a57r5 is "Room 5".
Understand "Area 57 in Room 5" as a57r5.
the parent of the a57r5 is r5.
the X of the a57r5 is "547564.0".
the Y of the a57r5 is "1435314.0".

[create a58r5]
a58r5 is a area. "An area (58) in r5".
the printed name of the a58r5 is "Room 5".
Understand "Area 58 in Room 5" as a58r5.
the parent of the a58r5 is r5.
the X of the a58r5 is "547532.9123931625".
the Y of the a58r5 is "1435288.1662393163".

[create a61r5]
a61r5 is a area. "An area (61) in r5".
the printed name of the a61r5 is "Room 5".
Understand "Area 61 in Room 5" as a61r5.
the parent of the a61r5 is r5.
the X of the a61r5 is "547333.0".
the Y of the a61r5 is "1435304.0".

[create a62r5]
a62r5 is a area. "An area (62) in r5".
the printed name of the a62r5 is "Room 5".
Understand "Area 62 in Room 5" as a62r5.
the parent of the a62r5 is r5.
the X of the a62r5 is "547300.005982906".
the Y of the a62r5 is "1435305.7505982905".

[create a75r5]
a75r5 is a area. "An area (75) in r5".
the printed name of the a75r5 is "Room 5".
Understand "Area 75 in Room 5" as a75r5.
the parent of the a75r5 is r5.
the X of the a75r5 is "547573.0".
the Y of the a75r5 is "1435224.0".

[create a76r5]
a76r5 is a area. "An area (76) in r5".
the printed name of the a76r5 is "Room 5".
Understand "Area 76 in Room 5" as a76r5.
the parent of the a76r5 is r5.
the X of the a76r5 is "547530.94".
the Y of the a76r5 is "1435242.519".

[create a79r6]
a79r6 is a area. "An area (79) in r6".
the printed name of the a79r6 is "Room 6".
Understand "Area 79 in Room 6" as a79r6.
the parent of the a79r6 is r6.
the X of the a79r6 is "550239.0".
the Y of the a79r6 is "1435160.31147541".

[create a80r6]
a80r6 is a area. "An area (80) in r6".
the printed name of the a80r6 is "Room 6".
Understand "Area 80 in Room 6" as a80r6.
the parent of the a80r6 is r6.
the X of the a80r6 is "550205.8444444444".
the Y of the a80r6 is "1435197.0".

[create d0]
d0 is a door. "A door between a18r3 and a22r4".
d0 is south of a18r3 and north of a22r4.

[create d2]
d2 is a door. "A door between a25r1 and a18r3".
d2 is northwest of a25r1 and southeast of a18r3.

[create d4]
d4 is a door. "A door between a26r1 and a3r5".
d4 is north of a26r1 and south of a3r5.

[create d6]
d6 is a door. "A door between a13r0 and a20r1".
d6 is south of a13r0 and north of a20r1.

[create d8]
d8 is a door. "A door between a12r1 and a11r2".
d8 is north of a12r1 and south of a11r2.

southeast of a0r0 is northwest of a1r0.

southeast of a2r5 is northwest of a3r5.

northeast of a3r5 is southwest of a4r5.

east of a4r5 is west of a5r5.

east of a6r3 is west of a7r3.

east of a8r0 is west of a9r0.

west of a0r0 is east of a9r0.

west of a2r5 is east of a10r5.

north of a13r0 is south of a14r0.

north of a11r2 is south of a15r2.

northeast of a14r0 is southwest of a16r0.

south of a9r0 is north of a14r0.

northeast of a16r0 is southwest of a17r0.

south of a7r3 is north of a18r3.

east of a17r0 is west of a19r0.

southeast of a20r1 is northwest of a21r1.

northeast of a23r1 is southwest of a24r1.

north of a24r1 is south of a25r1.

northeast of a25r1 is southwest of a26r1.

east of a22r4 is west of a27r4.

west of a22r4 is east of a28r4.

southwest of a12r1 is northeast of a21r1.

east of a26r1 is west of a29r1.

southeast of a24r1 is northwest of a30r1.

southwest of a20r1 is northeast of a29r1.

northwest of a31r4 is southeast of a32r4.

northeast of a33r4 is southwest of a34r4.

southwest of a35r4 is northeast of a36r4.

southeast of a37r4 is northwest of a38r4.

northwest of a39r1 is southeast of a40r1.

northeast of a41r1 is southwest of a42r1.

west of a43r1 is east of a44r1.

east of a45r5 is west of a46r5.

northeast of a47r5 is southwest of a48r5.

northeast of a49r0 is southwest of a50r0.

southeast of a51r5 is northwest of a52r5.

southwest of a53r0 is northeast of a54r0.

southeast of a55r3 is northwest of a56r3.

southwest of a57r5 is northeast of a58r5.

southeast of a59r0 is northwest of a60r0.

west of a61r5 is east of a62r5.

southeast of a63r0 is northwest of a64r0.

southwest of a65r2 is northeast of a66r2.

northeast of a67r0 is southwest of a68r0.

southeast of a69r0 is northwest of a70r0.

southwest of a71r3 is northeast of a72r3.

northeast of a73r3 is southwest of a74r3.

northwest of a75r5 is southeast of a76r5.

southeast of a30r1 is northwest of a77r1.

south of a23r1 is north of a78r1.

north of a15r2 is south of a79r2.

southeast of a19r0 is northwest of a80r0.

east of a1r0 is west of a69r0.

east of a11r2 is west of a66r2.

northwest of a16r0 is southeast of a68r0.

northwest of a17r0 is southeast of a70r0.

northeast of a7r3 is southwest of a72r3.

southwest of a6r3 is northeast of a74r3.

southeast of a5r5 is northwest of a76r5.

southwest of a10r5 is northeast of a48r5.

southwest of a8r0 is northeast of a50r0.

northwest of a10r5 is southeast of a52r5.

southeast of a27r4 is northwest of a32r4.

southwest of a28r4 is northeast of a34r4.

northeast of a27r4 is southwest of a36r4.

northwest of a28r4 is southeast of a38r4.

southeast of a29r1 is northwest of a40r1.

southwest of a21r1 is northeast of a42r1.

east of a12r1 is west of a44r1.

west of a13r0 is east of a64r0.

northeast of a0r0 is southwest of a54r0.

northwest of a6r3 is southeast of a56r3.

northeast of a5r5 is southwest of a58r5.

northwest of a8r0 is southeast of a60r0.

north of a4r5 is south of a62r5.

north of a2r5 is south of a46r5.

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
the player is in a56r3.

the orientation of the player is 6.
