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

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "546559.5080993522".
the Y of the a6r0 is "1625264.2282217424".

[create a7r0]
a7r0 is a area. "An area (7) in r0".
the printed name of the a7r0 is "Room 0".
Understand "Area 7 in Room 0" as a7r0.
the parent of the a7r0 is r0.
the X of the a7r0 is "546913.8704484985".
the Y of the a7r0 is "1625176.4146870642".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "547231.1865386873".
the Y of the a9r0 is "1625259.6370670423".

[create a38r0]
a38r0 is a area. "An area (38) in r0".
the printed name of the a38r0 is "Room 0".
Understand "Area 38 in Room 0" as a38r0.
the parent of the a38r0 is r0.
the X of the a38r0 is "546514.0".
the Y of the a38r0 is "1625314.0".

[create a39r0]
a39r0 is a area. "An area (39) in r0".
the printed name of the a39r0 is "Room 0".
Understand "Area 39 in Room 0" as a39r0.
the parent of the a39r0 is r0.
the X of the a39r0 is "546545.0876068377".
the Y of the a39r0 is "1625288.1662393163".

[create a42r0]
a42r0 is a area. "An area (42) in r0".
the printed name of the a42r0 is "Room 0".
Understand "Area 42 in Room 0" as a42r0.
the parent of the a42r0 is r0.
the X of the a42r0 is "546505.0".
the Y of the a42r0 is "1625224.0".

[create a43r0]
a43r0 is a area. "An area (43) in r0".
the printed name of the a43r0 is "Room 0".
Understand "Area 43 in Room 0" as a43r0.
the parent of the a43r0 is r0.
the X of the a43r0 is "546532.2059471366".
the Y of the a43r0 is "1625244.0044052866".

[create a46r0]
a46r0 is a area. "An area (46) in r0".
the printed name of the a46r0 is "Room 0".
Understand "Area 46 in Room 0" as a46r0.
the parent of the a46r0 is r0.
the X of the a46r0 is "547279.0".
the Y of the a46r0 is "1625314.0".

[create a47r0]
a47r0 is a area. "An area (47) in r0".
the printed name of the a47r0 is "Room 0".
Understand "Area 47 in Room 0" as a47r0.
the parent of the a47r0 is r0.
the X of the a47r0 is "547248.1050372066".
the Y of the a47r0 is "1625287.7894485786".

[create a62r0]
a62r0 is a area. "An area (62) in r0".
the printed name of the a62r0 is "Room 0".
Understand "Area 62 in Room 0" as a62r0.
the parent of the a62r0 is r0.
the X of the a62r0 is "547289.0".
the Y of the a62r0 is "1625224.0".

[create a63r0]
a63r0 is a area. "An area (63) in r0".
the printed name of the a63r0 is "Room 0".
Understand "Area 63 in Room 0" as a63r0.
the parent of the a63r0 is r0.
the X of the a63r0 is "547246.1798056157".
the Y of the a63r0 is "1625242.019978402".

[create a13r1]
a13r1 is a area. "An area (13) in r1".
the printed name of the a13r1 is "Room 1".
Understand "Area 13 in Room 1" as a13r1.
the parent of the a13r1 is r1.
the X of the a13r1 is "546821.0879044278".
the Y of the a13r1 is "1623574.8019164002".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "546888.8508562942".
the Y of the a14r1 is "1623657.0695365253".

[create a15r1]
a15r1 is a area. "An area (15) in r1".
the printed name of the a15r1 is "Room 1".
Understand "Area 15 in Room 1" as a15r1.
the parent of the a15r1 is r1.
the X of the a15r1 is "547117.5948465099".
the Y of the a15r1 is "1623739.023855546".

[create a16r1]
a16r1 is a area. "An area (16) in r1".
the printed name of the a16r1 is "Room 1".
Understand "Area 16 in Room 1" as a16r1.
the parent of the a16r1 is r1.
the X of the a16r1 is "547225.2139041875".
the Y of the a16r1 is "1624043.1570571826".

[create a17r1]
a17r1 is a area. "An area (17) in r1".
the printed name of the a17r1 is "Room 1".
Understand "Area 17 in Room 1" as a17r1.
the parent of the a17r1 is r1.
the X of the a17r1 is "548093.9076679607".
the Y of the a17r1 is "1624056.9294065498".

[create a23r1]
a23r1 is a area. "An area (23) in r1".
the printed name of the a23r1 is "Room 1".
Understand "Area 23 in Room 1" as a23r1.
the parent of the a23r1 is r1.
the X of the a23r1 is "547004.3961068327".
the Y of the a23r1 is "1623328.7338161522".

[create a24r1]
a24r1 is a area. "An area (24) in r1".
the printed name of the a24r1 is "Room 1".
Understand "Area 24 in Room 1" as a24r1.
the parent of the a24r1 is r1.
the X of the a24r1 is "547050.7945498944".
the Y of the a24r1 is "1623604.0981092223".

[create a25r1]
a25r1 is a area. "An area (25) in r1".
the printed name of the a25r1 is "Room 1".
Understand "Area 25 in Room 1" as a25r1.
the parent of the a25r1 is r1.
the X of the a25r1 is "546683.2565472326".
the Y of the a25r1 is "1623385.1937898204".

[create a26r1]
a26r1 is a area. "An area (26) in r1".
the printed name of the a26r1 is "Room 1".
Understand "Area 26 in Room 1" as a26r1.
the parent of the a26r1 is r1.
the X of the a26r1 is "548330.4310969368".
the Y of the a26r1 is "1623367.4370029136".

[create a27r1]
a27r1 is a area. "An area (27) in r1".
the printed name of the a27r1 is "Room 1".
Understand "Area 27 in Room 1" as a27r1.
the parent of the a27r1 is r1.
the X of the a27r1 is "545513.8224172432".
the Y of the a27r1 is "1623557.5305011156".

[create a28r1]
a28r1 is a area. "An area (28) in r1".
the printed name of the a28r1 is "Room 1".
Understand "Area 28 in Room 1" as a28r1.
the parent of the a28r1 is r1.
the X of the a28r1 is "547949.6344696969".
the Y of the a28r1 is "1623244.0018939392".

[create a29r1]
a29r1 is a area. "An area (29) in r1".
the printed name of the a29r1 is "Room 1".
Understand "Area 29 in Room 1" as a29r1.
the parent of the a29r1 is r1.
the X of the a29r1 is "547341.6659021408".
the Y of the a29r1 is "1623243.50127421".

[create a30r1]
a30r1 is a area. "An area (30) in r1".
the printed name of the a30r1 is "Room 1".
Understand "Area 30 in Room 1" as a30r1.
the parent of the a30r1 is r1.
the X of the a30r1 is "548696.7901234569".
the Y of the a30r1 is "1623243.2942386833".

[create a31r1]
a31r1 is a area. "An area (31) in r1".
the printed name of the a31r1 is "Room 1".
Understand "Area 31 in Room 1" as a31r1.
the parent of the a31r1 is r1.
the X of the a31r1 is "546668.1365448317".
the Y of the a31r1 is "1623255.7930181918".

[create a32r1]
a32r1 is a area. "An area (32) in r1".
the printed name of the a32r1 is "Room 1".
Understand "Area 32 in Room 1" as a32r1.
the parent of the a32r1 is r1.
the X of the a32r1 is "546539.1682779246".
the Y of the a32r1 is "1623284.3417026664".

[create a33r1]
a33r1 is a area. "An area (33) in r1".
the printed name of the a33r1 is "Room 1".
Understand "Area 33 in Room 1" as a33r1.
the parent of the a33r1 is r1.
the X of the a33r1 is "546600.9256228264".
the Y of the a33r1 is "1623317.4544145265".

[create a58r1]
a58r1 is a area. "An area (58) in r1".
the printed name of the a58r1 is "Room 1".
Understand "Area 58 in Room 1" as a58r1.
the parent of the a58r1 is r1.
the X of the a58r1 is "545347.0".
the Y of the a58r1 is "1623473.0".

[create a59r1]
a59r1 is a area. "An area (59) in r1".
the printed name of the a59r1 is "Room 1".
Understand "Area 59 in Room 1" as a59r1.
the parent of the a59r1 is r1.
the X of the a59r1 is "545417.9942396786".
the Y of the a59r1 is "1623511.6157052463".

[create a72r1]
a72r1 is a area. "An area (72) in r1".
the printed name of the a72r1 is "Room 1".
Understand "Area 72 in Room 1" as a72r1.
the parent of the a72r1 is r1.
the X of the a72r1 is "548744.0".
the Y of the a72r1 is "1623191.0".

[create a73r1]
a73r1 is a area. "An area (73) in r1".
the printed name of the a73r1 is "Room 1".
Understand "Area 73 in Room 1" as a73r1.
the parent of the a73r1 is r1.
the X of the a73r1 is "548712.7862838916".
the Y of the a73r1 is "1623216.6339712918".

[create a74r1]
a74r1 is a area. "An area (74) in r1".
the printed name of the a74r1 is "Room 1".
Understand "Area 74 in Room 1" as a74r1.
the parent of the a74r1 is r1.
the X of the a74r1 is "547389.0".
the Y of the a74r1 is "1623191.0".

[create a75r1]
a75r1 is a area. "An area (75) in r1".
the printed name of the a75r1 is "Room 1".
Understand "Area 75 in Room 1" as a75r1.
the parent of the a75r1 is r1.
the X of the a75r1 is "547357.7862838915".
the Y of the a75r1 is "1623216.6339712918".

[create a76r1]
a76r1 is a area. "An area (76) in r1".
the printed name of the a76r1 is "Room 1".
Understand "Area 76 in Room 1" as a76r1.
the parent of the a76r1 is r1.
the X of the a76r1 is "547902.0".
the Y of the a76r1 is "1623191.0".

[create a77r1]
a77r1 is a area. "An area (77) in r1".
the printed name of the a77r1 is "Room 1".
Understand "Area 77 in Room 1" as a77r1.
the parent of the a77r1 is r1.
the X of the a77r1 is "547933.2137161085".
the Y of the a77r1 is "1623216.6339712918".

[create a78r1]
a78r1 is a area. "An area (78) in r1".
the printed name of the a78r1 is "Room 1".
Understand "Area 78 in Room 1" as a78r1.
the parent of the a78r1 is r1.
the X of the a78r1 is "548753.0".
the Y of the a78r1 is "1623280.0".

[create a79r1]
a79r1 is a area. "An area (79) in r1".
the printed name of the a79r1 is "Room 1".
Understand "Area 79 in Room 1" as a79r1.
the parent of the a79r1 is r1.
the X of the a79r1 is "548713.3329344183".
the Y of the a79r1 is "1623261.5337482048".

[create a80r1]
a80r1 is a area. "An area (80) in r1".
the printed name of the a80r1 is "Room 1".
Understand "Area 80 in Room 1" as a80r1.
the parent of the a80r1 is r1.
the X of the a80r1 is "547398.0".
the Y of the a80r1 is "1623280.0".

[create a81r1]
a81r1 is a area. "An area (81) in r1".
the printed name of the a81r1 is "Room 1".
Understand "Area 81 in Room 1" as a81r1.
the parent of the a81r1 is r1.
the X of the a81r1 is "547357.7054722286".
the Y of the a81r1 is "1623261.5971994374".

[create a82r1]
a82r1 is a area. "An area (82) in r1".
the printed name of the a82r1 is "Room 1".
Understand "Area 82 in Room 1" as a82r1.
the parent of the a82r1 is r1.
the X of the a82r1 is "547893.0".
the Y of the a82r1 is "1623280.0".

[create a83r1]
a83r1 is a area. "An area (83) in r1".
the printed name of the a83r1 is "Room 1".
Understand "Area 83 in Room 1" as a83r1.
the parent of the a83r1 is r1.
the X of the a83r1 is "547934.3556858188".
the Y of the a83r1 is "1623261.7045075544".

[create a84r1]
a84r1 is a area. "An area (84) in r1".
the printed name of the a84r1 is "Room 1".
Understand "Area 84 in Room 1" as a84r1.
the parent of the a84r1 is r1.
the X of the a84r1 is "546630.0".
the Y of the a84r1 is "1623191.0".

[create a85r1]
a85r1 is a area. "An area (85) in r1".
the printed name of the a85r1 is "Room 1".
Understand "Area 85 in Room 1" as a85r1.
the parent of the a85r1 is r1.
the X of the a85r1 is "546662.0154884118".
the Y of the a85r1 is "1623199.351323865".

[create a10r2]
a10r2 is a area. "An area (10) in r2".
the printed name of the a10r2 is "Room 2".
Understand "Area 10 in Room 2" as a10r2.
the parent of the a10r2 is r2.
the X of the a10r2 is "547623.5280936206".
the Y of the a10r2 is "1625164.217310295".

[create a11r2]
a11r2 is a area. "An area (11) in r2".
the printed name of the a11r2 is "Room 2".
Understand "Area 11 in Room 2" as a11r2.
the parent of the a11r2 is r2.
the X of the a11r2 is "547675.7557274038".
the Y of the a11r2 is "1625163.3385288818".

[create a12r2]
a12r2 is a area. "An area (12) in r2".
the printed name of the a12r2 is "Room 2".
Understand "Area 12 in Room 2" as a12r2.
the parent of the a12r2 is r2.
the X of the a12r2 is "547748.2593157241".
the Y of the a12r2 is "1625279.5971839244".

[create a44r2]
a44r2 is a area. "An area (44) in r2".
the printed name of the a44r2 is "Room 2".
Understand "Area 44 in Room 2" as a44r2.
the parent of the a44r2 is r2.
the X of the a44r2 is "547509.0".
the Y of the a44r2 is "1625301.0".

[create a45r2]
a45r2 is a area. "An area (45) in r2".
the printed name of the a45r2 is "Room 2".
Understand "Area 45 in Room 2" as a45r2.
the parent of the a45r2 is r2.
the X of the a45r2 is "547549.1349339068".
the Y of the a45r2 is "1625246.6523434473".

[create a70r2]
a70r2 is a area. "An area (70) in r2".
the printed name of the a70r2 is "Room 2".
Understand "Area 70 in Room 2" as a70r2.
the parent of the a70r2 is r2.
the X of the a70r2 is "547776.0".
the Y of the a70r2 is "1625332.0".

[create a0r3]
a0r3 is a area. "An area (0) in r3".
the printed name of the a0r3 is "Room 3".
Understand "Area 0 in Room 3" as a0r3.
the parent of the a0r3 is r3.
the X of the a0r3 is "545125.2443992454".
the Y of the a0r3 is "1625259.5559191133".

[create a1r3]
a1r3 is a area. "An area (1) in r3".
the printed name of the a1r3 is "Room 3".
Understand "Area 1 in Room 3" as a1r3.
the parent of the a1r3 is r3.
the X of the a1r3 is "545485.0".
the Y of the a1r3 is "1625137.7083333333".

[create a19r3]
a19r3 is a area. "An area (19) in r3".
the printed name of the a19r3 is "Room 3".
Understand "Area 19 in Room 3" as a19r3.
the parent of the a19r3 is r3.
the X of the a19r3 is "545839.8606168446".
the Y of the a19r3 is "1625259.8602906289".

[create a50r3]
a50r3 is a area. "An area (50) in r3".
the printed name of the a50r3 is "Room 3".
Understand "Area 50 in Room 3" as a50r3.
the parent of the a50r3 is r3.
the X of the a50r3 is "545077.0".
the Y of the a50r3 is "1625314.0".

[create a51r3]
a51r3 is a area. "An area (51) in r3".
the printed name of the a51r3 is "Room 3".
Understand "Area 51 in Room 3" as a51r3.
the parent of the a51r3 is r3.
the X of the a51r3 is "545107.9153780069".
the Y of the a51r3 is "1625288.1834621995".

[create a54r3]
a54r3 is a area. "An area (54) in r3".
the printed name of the a54r3 is "Room 3".
Understand "Area 54 in Room 3" as a54r3.
the parent of the a54r3 is r3.
the X of the a54r3 is "545888.0".
the Y of the a54r3 is "1625314.0".

[create a55r3]
a55r3 is a area. "An area (55) in r3".
the printed name of the a55r3 is "Room 3".
Understand "Area 55 in Room 3" as a55r3.
the parent of the a55r3 is r3.
the X of the a55r3 is "545856.9123931624".
the Y of the a55r3 is "1625288.1662393163".

[create a60r3]
a60r3 is a area. "An area (60) in r3".
the printed name of the a60r3 is "Room 3".
Understand "Area 60 in Room 3" as a60r3.
the parent of the a60r3 is r3.
the X of the a60r3 is "545068.0".
the Y of the a60r3 is "1625224.0".

[create a61r3]
a61r3 is a area. "An area (61) in r3".
the printed name of the a61r3 is "Room 3".
Understand "Area 61 in Room 3" as a61r3.
the parent of the a61r3 is r3.
the X of the a61r3 is "545110.6911764706".
the Y of the a61r3 is "1625242.4558823528".

[create a64r3]
a64r3 is a area. "An area (64) in r3".
the printed name of the a64r3 is "Room 3".
Understand "Area 64 in Room 3" as a64r3.
the parent of the a64r3 is r3.
the X of the a64r3 is "545897.0".
the Y of the a64r3 is "1625224.0".

[create a65r3]
a65r3 is a area. "An area (65) in r3".
the printed name of the a65r3 is "Room 3".
Understand "Area 65 in Room 3" as a65r3.
the parent of the a65r3 is r3.
the X of the a65r3 is "545854.94".
the Y of the a65r3 is "1625242.519".

[create a4r4]
a4r4 is a area. "An area (4) in r4".
the printed name of the a4r4 is "Room 4".
Understand "Area 4 in Room 4" as a4r4.
the parent of the a4r4 is r4.
the X of the a4r4 is "549564.54050483".
the Y of the a4r4 is "1625260.8544286496".

[create a5r4]
a5r4 is a area. "An area (5) in r4".
the printed name of the a5r4 is "Room 4".
Understand "Area 5 in Room 4" as a5r4.
the parent of the a5r4 is r4.
the X of the a5r4 is "549904.4895726717".
the Y of the a5r4 is "1625172.4549042068".

[create a8r4]
a8r4 is a area. "An area (8) in r4".
the printed name of the a8r4 is "Room 4".
Understand "Area 8 in Room 4" as a8r4.
the parent of the a8r4 is r4.
the X of the a8r4 is "550228.8307569795".
the Y of the a8r4 is "1625261.4707232525".

[create a36r4]
a36r4 is a area. "An area (36) in r4".
the printed name of the a36r4 is "Room 4".
Understand "Area 36 in Room 4" as a36r4.
the parent of the a36r4 is r4.
the X of the a36r4 is "549517.0".
the Y of the a36r4 is "1625314.0".

[create a37r4]
a37r4 is a area. "An area (37) in r4".
the printed name of the a37r4 is "Room 4".
Understand "Area 37 in Room 4" as a37r4.
the parent of the a37r4 is r4.
the X of the a37r4 is "549548.0876068377".
the Y of the a37r4 is "1625288.1662393163".

[create a40r4]
a40r4 is a area. "An area (40) in r4".
the printed name of the a40r4 is "Room 4".
Understand "Area 40 in Room 4" as a40r4.
the parent of the a40r4 is r4.
the X of the a40r4 is "550285.0".
the Y of the a40r4 is "1625224.0".

[create a41r4]
a41r4 is a area. "An area (41) in r4".
the printed name of the a41r4 is "Room 4".
Understand "Area 41 in Room 4" as a41r4.
the parent of the a41r4 is r4.
the X of the a41r4 is "550246.1323529411".
the Y of the a41r4 is "1625242.8382352942".

[create a52r4]
a52r4 is a area. "An area (52) in r4".
the printed name of the a52r4 is "Room 4".
Understand "Area 52 in Room 4" as a52r4.
the parent of the a52r4 is r4.
the X of the a52r4 is "550276.0".
the Y of the a52r4 is "1625314.0".

[create a53r4]
a53r4 is a area. "An area (53) in r4".
the printed name of the a53r4 is "Room 4".
Understand "Area 53 in Room 4" as a53r4.
the parent of the a53r4 is r4.
the X of the a53r4 is "550244.9123931625".
the Y of the a53r4 is "1625288.1662393163".

[create a68r4]
a68r4 is a area. "An area (68) in r4".
the printed name of the a68r4 is "Room 4".
Understand "Area 68 in Room 4" as a68r4.
the parent of the a68r4 is r4.
the X of the a68r4 is "549508.0".
the Y of the a68r4 is "1625224.0".

[create a69r4]
a69r4 is a area. "An area (69) in r4".
the printed name of the a69r4 is "Room 4".
Understand "Area 69 in Room 4" as a69r4.
the parent of the a69r4 is r4.
the X of the a69r4 is "549548.7622699386".
the Y of the a69r4 is "1625242.648773006".

[create a20r5]
a20r5 is a area. "An area (20) in r5".
the printed name of the a20r5 is "Room 5".
Understand "Area 20 in Room 5" as a20r5.
the parent of the a20r5 is r5.
the X of the a20r5 is "549259.9629832632".
the Y of the a20r5 is "1623328.2779362".

[create a21r5]
a21r5 is a area. "An area (21) in r5".
the printed name of the a21r5 is "Room 5".
Understand "Area 21 in Room 5" as a21r5.
the parent of the a21r5 is r5.
the X of the a21r5 is "549098.0613114522".
the Y of the a21r5 is "1623224.5596518875".

[create a22r5]
a22r5 is a area. "An area (22) in r5".
the printed name of the a22r5 is "Room 5".
Understand "Area 22 in Room 5" as a22r5.
the parent of the a22r5 is r5.
the X of the a22r5 is "549168.6294029708".
the Y of the a22r5 is "1623333.519746642".

[create a56r5]
a56r5 is a area. "An area (56) in r5".
the printed name of the a56r5 is "Room 5".
Understand "Area 56 in Room 5" as a56r5.
the parent of the a56r5 is r5.
the X of the a56r5 is "549366.0".
the Y of the a56r5 is "1623200.0".

[create a57r5]
a57r5 is a area. "An area (57) in r5".
the printed name of the a57r5 is "Room 5".
Understand "Area 57 in Room 5" as a57r5.
the parent of the a57r5 is r5.
the X of the a57r5 is "549323.4517163367".
the Y of the a57r5 is "1623249.4252993807".

[create a86r5]
a86r5 is a area. "An area (86) in r5".
the printed name of the a86r5 is "Room 5".
Understand "Area 86 in Room 5" as a86r5.
the parent of the a86r5 is r5.
the X of the a86r5 is "549065.0".
the Y of the a86r5 is "1623172.0".

[create a2r6]
a2r6 is a area. "An area (2) in r6".
the printed name of the a2r6 is "Room 6".
Understand "Area 2 in Room 6" as a2r6.
the parent of the a2r6 is r6.
the X of the a2r6 is "548553.6942110015".
the Y of the a2r6 is "1625148.1635622133".

[create a3r6]
a3r6 is a area. "An area (3) in r6".
the printed name of the a3r6 is "Room 6".
Understand "Area 3 in Room 6" as a3r6.
the parent of the a3r6 is r6.
the X of the a3r6 is "548922.692765556".
the Y of the a3r6 is "1625261.1041153665".

[create a18r6]
a18r6 is a area. "An area (18) in r6".
the printed name of the a18r6 is "Room 6".
Understand "Area 18 in Room 6" as a18r6.
the parent of the a18r6 is r6.
the X of the a18r6 is "548203.281371821".
the Y of the a18r6 is "1625261.2845894438".

[create a34r6]
a34r6 is a area. "An area (34) in r6".
the printed name of the a34r6 is "Room 6".
Understand "Area 34 in Room 6" as a34r6.
the parent of the a34r6 is r6.
the X of the a34r6 is "548156.0".
the Y of the a34r6 is "1625314.0".

[create a35r6]
a35r6 is a area. "An area (35) in r6".
the printed name of the a35r6 is "Room 6".
Understand "Area 35 in Room 6" as a35r6.
the parent of the a35r6 is r6.
the X of the a35r6 is "548187.0876068376".
the Y of the a35r6 is "1625288.1662393163".

[create a48r6]
a48r6 is a area. "An area (48) in r6".
the printed name of the a48r6 is "Room 6".
Understand "Area 48 in Room 6" as a48r6.
the parent of the a48r6 is r6.
the X of the a48r6 is "548970.0".
the Y of the a48r6 is "1625314.0".

[create a49r6]
a49r6 is a area. "An area (49) in r6".
the printed name of the a49r6 is "Room 6".
Understand "Area 49 in Room 6" as a49r6.
the parent of the a49r6 is r6.
the X of the a49r6 is "548939.0846219934".
the Y of the a49r6 is "1625288.1834621993".

[create a66r6]
a66r6 is a area. "An area (66) in r6".
the printed name of the a66r6 is "Room 6".
Understand "Area 66 in Room 6" as a66r6.
the parent of the a66r6 is r6.
the X of the a66r6 is "548979.0".
the Y of the a66r6 is "1625224.0".

[create a67r6]
a67r6 is a area. "An area (67) in r6".
the printed name of the a67r6 is "Room 6".
Understand "Area 67 in Room 6" as a67r6.
the parent of the a67r6 is r6.
the X of the a67r6 is "548938.5982564336".
the Y of the a67r6 is "1625242.6848256432".

[create a71r6]
a71r6 is a area. "An area (71) in r6".
the printed name of the a71r6 is "Room 6".
Understand "Area 71 in Room 6" as a71r6.
the parent of the a71r6 is r6.
the X of the a71r6 is "548186.4941364606".
the Y of the a71r6 is "1625242.775586354".

[create d0]
d0 is a door. "A door between a5r4 and a20r5".
d0 is south of a5r4 and north of a20r5.

[create d2]
d2 is a door. "A door between a22r5 and a2r6".
d2 is north of a22r5 and south of a2r6.

[create d4]
d4 is a door. "A door between a27r1 and a1r3".
d4 is north of a27r1 and south of a1r3.

[create d6]
d6 is a door. "A door between a7r0 and a14r1".
d6 is south of a7r0 and north of a14r1.

[create d8]
d8 is a door. "A door between a17r1 and a11r2".
d8 is north of a17r1 and south of a11r2.

[create d9]
d9 is a door. "A door between a16r1 and a10r2".
d9 is north of a16r1 and south of a10r2.

east of a0r3 is west of a1r3.

east of a2r6 is west of a3r6.

east of a4r4 is west of a5r4.

east of a6r0 is west of a7r0.

east of a5r4 is west of a8r4.

east of a7r0 is west of a9r0.

east of a10r2 is west of a11r2.

northeast of a11r2 is southwest of a12r2.

northeast of a13r1 is southwest of a14r1.

north of a15r1 is south of a16r1.

east of a16r1 is west of a17r1.

west of a2r6 is east of a18r6.

east of a1r3 is west of a19r3.

northeast of a21r5 is southwest of a22r5.

north of a23r1 is south of a24r1.

southwest of a13r1 is northeast of a25r1.

south of a17r1 is north of a26r1.

southwest of a15r1 is northeast of a24r1.

west of a26r1 is east of a28r1.

east of a23r1 is west of a29r1.

east of a26r1 is west of a30r1.

west of a23r1 is east of a31r1.

east of a27r1 is west of a32r1.

northeast of a32r1 is southwest of a33r1.

southwest of a25r1 is northeast of a33r1.

west of a20r5 is east of a22r5.

southeast of a34r6 is northwest of a35r6.

southeast of a36r4 is northwest of a37r4.

southeast of a38r0 is northwest of a39r0.

northwest of a40r4 is southeast of a41r4.

northeast of a42r0 is southwest of a43r0.

southeast of a44r2 is northwest of a45r2.

southwest of a46r0 is northeast of a47r0.

southwest of a48r6 is northeast of a49r6.

southeast of a50r3 is northwest of a51r3.

southwest of a52r4 is northeast of a53r4.

southwest of a54r3 is northeast of a55r3.

northwest of a56r5 is southeast of a57r5.

northeast of a58r1 is southwest of a59r1.

northeast of a60r3 is southwest of a61r3.

northwest of a62r0 is southeast of a63r0.

northwest of a64r3 is southeast of a65r3.

northwest of a66r6 is southeast of a67r6.

northeast of a68r4 is southwest of a69r4.

northwest of a72r1 is southeast of a73r1.

northwest of a74r1 is southeast of a75r1.

northeast of a76r1 is southwest of a77r1.

southwest of a78r1 is northeast of a79r1.

southwest of a80r1 is northeast of a81r1.

southeast of a82r1 is northwest of a83r1.

east of a84r1 is west of a85r1.

southwest of a21r5 is northeast of a86r5.

northeast of a12r2 is southwest of a70r2.

northeast of a29r1 is southwest of a81r1.

northwest of a28r1 is southeast of a83r1.

south of a31r1 is north of a85r1.

northwest of a31r1 is southeast of a33r1.

southeast of a20r5 is northwest of a57r5.

southwest of a27r1 is northeast of a59r1.

east of a14r1 is west of a24r1.

southwest of a0r3 is northeast of a61r3.

southeast of a9r0 is northwest of a63r0.

southeast of a30r1 is northwest of a73r1.

southeast of a29r1 is northwest of a75r1.

southwest of a28r1 is northeast of a77r1.

northeast of a30r1 is southwest of a79r1.

northeast of a9r0 is southwest of a47r0.

northeast of a3r6 is southwest of a49r6.

northwest of a0r3 is southeast of a51r3.

northeast of a8r4 is southwest of a53r4.

northeast of a19r3 is southwest of a55r3.

northwest of a18r6 is southeast of a35r6.

northwest of a4r4 is southeast of a37r4.

northwest of a6r0 is southeast of a39r0.

southeast of a19r3 is northwest of a65r3.

southeast of a3r6 is northwest of a67r6.

southwest of a4r4 is northeast of a69r4.

southwest of a18r6 is northeast of a71r6.

southeast of a8r4 is northwest of a41r4.

southwest of a6r0 is northeast of a43r0.

northwest of a10r2 is southeast of a45r2.

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
the player is in a23r1.

the orientation of the player is 6.
