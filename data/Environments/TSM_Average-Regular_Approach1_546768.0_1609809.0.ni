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

[create a0r0]
a0r0 is a area. "An area (0) in r0".
the printed name of the a0r0 is "Room 0".
Understand "Area 0 in Room 0" as a0r0.
the parent of the a0r0 is r0.
the X of the a0r0 is "547889.0".
the Y of the a0r0 is "1607991.0".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "547857.9579422104".
the Y of the a1r0 is "1608016.6166125853".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "547413.0".
the Y of the a22r0 is "1607969.0".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "547374.1111111111".
the Y of the a23r0 is "1608044.0960188392".

[create a26r0]
a26r0 is a area. "An area (26) in r0".
the printed name of the a26r0 is "Room 0".
Understand "Area 26 in Room 0" as a26r0.
the parent of the a26r0 is r0.
the X of the a26r0 is "546501.0".
the Y of the a26r0 is "1608531.0".

[create a27r0]
a27r0 is a area. "An area (27) in r0".
the printed name of the a27r0 is "Room 0".
Understand "Area 27 in Room 0" as a27r0.
the parent of the a27r0 is r0.
the X of the a27r0 is "546540.6586345383".
the Y of the a27r0 is "1608497.8333333333".

[create a28r0]
a28r0 is a area. "An area (28) in r0".
the printed name of the a28r0 is "Room 0".
Understand "Area 28 in Room 0" as a28r0.
the parent of the a28r0 is r0.
the X of the a28r0 is "546584.0".
the Y of the a28r0 is "1608568.0".

[create a29r0]
a29r0 is a area. "An area (29) in r0".
the printed name of the a29r0 is "Room 0".
Understand "Area 29 in Room 0" as a29r0.
the parent of the a29r0 is r0.
the X of the a29r0 is "546574.4164161289".
the Y of the a29r0 is "1608533.7010124677".

[create a30r0]
a30r0 is a area. "An area (30) in r0".
the printed name of the a30r0 is "Room 0".
Understand "Area 30 in Room 0" as a30r0.
the parent of the a30r0 is r0.
the X of the a30r0 is "547779.0".
the Y of the a30r0 is "1608310.0".

[create a31r0]
a31r0 is a area. "An area (31) in r0".
the printed name of the a31r0 is "Room 0".
Understand "Area 31 in Room 0" as a31r0.
the parent of the a31r0 is r0.
the X of the a31r0 is "547692.5351583213".
the Y of the a31r0 is "1608348.5".

[create a32r0]
a32r0 is a area. "An area (32) in r0".
the printed name of the a32r0 is "Room 0".
Understand "Area 32 in Room 0" as a32r0.
the parent of the a32r0 is r0.
the X of the a32r0 is "547779.0".
the Y of the a32r0 is "1608826.0".

[create a33r0]
a33r0 is a area. "An area (33) in r0".
the printed name of the a33r0 is "Room 0".
Understand "Area 33 in Room 0" as a33r0.
the parent of the a33r0 is r0.
the X of the a33r0 is "547714.0604806269".
the Y of the a33r0 is "1608790.375".

[create a40r0]
a40r0 is a area. "An area (40) in r0".
the printed name of the a40r0 is "Room 0".
Understand "Area 40 in Room 0" as a40r0.
the parent of the a40r0 is r0.
the X of the a40r0 is "547898.0".
the Y of the a40r0 is "1608080.0".

[create a41r0]
a41r0 is a area. "An area (41) in r0".
the printed name of the a41r0 is "Room 0".
Understand "Area 41 in Room 0" as a41r0.
the parent of the a41r0 is r0.
the X of the a41r0 is "547870.849158728".
the Y of the a41r0 is "1608060.2680626004".

[create a54r0]
a54r0 is a area. "An area (54) in r0".
the printed name of the a54r0 is "Room 0".
Understand "Area 54 in Room 0" as a54r0.
the parent of the a54r0 is r0.
the X of the a54r0 is "546451.0".
the Y of the a54r0 is "1608080.0".

[create a55r0]
a55r0 is a area. "An area (55) in r0".
the printed name of the a55r0 is "Room 0".
Understand "Area 55 in Room 0" as a55r0.
the parent of the a55r0 is r0.
the X of the a55r0 is "546499.428242099".
the Y of the a55r0 is "1608103.1313011707".

[create a70r0]
a70r0 is a area. "An area (70) in r0".
the printed name of the a70r0 is "Room 0".
Understand "Area 70 in Room 0" as a70r0.
the parent of the a70r0 is r0.
the X of the a70r0 is "547363.6509019042".
the Y of the a70r0 is "1608081.5673580447".

[create a73r0]
a73r0 is a area. "An area (73) in r0".
the printed name of the a73r0 is "Room 0".
Understand "Area 73 in Room 0" as a73r0.
the parent of the a73r0 is r0.
the X of the a73r0 is "546543.2405909916".
the Y of the a73r0 is "1608496.5702544397".

[create a74r0]
a74r0 is a area. "An area (74) in r0".
the printed name of the a74r0 is "Room 0".
Understand "Area 74 in Room 0" as a74r0.
the parent of the a74r0 is r0.
the X of the a74r0 is "547583.9816088659".
the Y of the a74r0 is "1608394.6303755324".

[create a75r0]
a75r0 is a area. "An area (75) in r0".
the printed name of the a75r0 is "Room 0".
Understand "Area 75 in Room 0" as a75r0.
the parent of the a75r0 is r0.
the X of the a75r0 is "547598.5909408364".
the Y of the a75r0 is "1608728.7152774585".

[create a78r0]
a78r0 is a area. "An area (78) in r0".
the printed name of the a78r0 is "Room 0".
Understand "Area 78 in Room 0" as a78r0.
the parent of the a78r0 is r0.
the X of the a78r0 is "547843.8228866081".
the Y of the a78r0 is "1608040.0615040062".

[create a82r0]
a82r0 is a area. "An area (82) in r0".
the printed name of the a82r0 is "Room 0".
Understand "Area 82 in Room 0" as a82r0.
the parent of the a82r0 is r0.
the X of the a82r0 is "546601.9453007282".
the Y of the a82r0 is "1608157.8516632349".

[create a83r0]
a83r0 is a area. "An area (83) in r0".
the printed name of the a83r0 is "Room 0".
Understand "Area 83 in Room 0" as a83r0.
the parent of the a83r0 is r0.
the X of the a83r0 is "547472.1503735131".
the Y of the a83r0 is "1608142.8738305636".

[create a92r0]
a92r0 is a area. "An area (92) in r0".
the printed name of the a92r0 is "Room 0".
Understand "Area 92 in Room 0" as a92r0.
the parent of the a92r0 is r0.
the X of the a92r0 is "547436.9494024159".
the Y of the a92r0 is "1608877.052784573".

[create a2r1]
a2r1 is a area. "An area (2) in r1".
the printed name of the a2r1 is "Room 1".
Understand "Area 2 in Room 1" as a2r1.
the parent of the a2r1 is r1.
the X of the a2r1 is "546835.0".
the Y of the a2r1 is "1610132.0".

[create a3r1]
a3r1 is a area. "An area (3) in r1".
the printed name of the a3r1 is "Room 1".
Understand "Area 3 in Room 1" as a3r1.
the parent of the a3r1 is r1.
the X of the a3r1 is "546870.375".
the Y of the a3r1 is "1610078.0838349436".

[create a4r1]
a4r1 is a area. "An area (4) in r1".
the printed name of the a4r1 is "Room 1".
Understand "Area 4 in Room 1" as a4r1.
the parent of the a4r1 is r1.
the X of the a4r1 is "546855.0".
the Y of the a4r1 is "1609588.0".

[create a5r1]
a5r1 is a area. "An area (5) in r1".
the printed name of the a5r1 is "Room 1".
Understand "Area 5 in Room 1" as a5r1.
the parent of the a5r1 is r1.
the X of the a5r1 is "546864.2785179659".
the Y of the a5r1 is "1609622.983324563".

[create a6r1]
a6r1 is a area. "An area (6) in r1".
the printed name of the a6r1 is "Room 1".
Understand "Area 6 in Room 1" as a6r1.
the parent of the a6r1 is r1.
the X of the a6r1 is "546938.0".
the Y of the a6r1 is "1609625.0".

[create a7r1]
a7r1 is a area. "An area (7) in r1".
the printed name of the a7r1 is "Room 1".
Understand "Area 7 in Room 1" as a7r1.
the parent of the a7r1 is r1.
the X of the a7r1 is "546900.0515730678".
the Y of the a7r1 is "1609654.3302550104".

[create a8r1]
a8r1 is a area. "An area (8) in r1".
the printed name of the a8r1 is "Room 1".
Understand "Area 8 in Room 1" as a8r1.
the parent of the a8r1 is r1.
the X of the a8r1 is "545034.0".
the Y of the a8r1 is "1609625.0".

[create a9r1]
a9r1 is a area. "An area (9) in r1".
the printed name of the a9r1 is "Room 1".
Understand "Area 9 in Room 1" as a9r1.
the parent of the a9r1 is r1.
the X of the a9r1 is "545071.8974585026".
the Y of the a9r1 is "1609654.215920425".

[create a10r1]
a10r1 is a area. "An area (10) in r1".
the printed name of the a10r1 is "Room 1".
Understand "Area 10 in Room 1" as a10r1.
the parent of the a10r1 is r1.
the X of the a10r1 is "545117.0".
the Y of the a10r1 is "1609588.0".

[create a11r1]
a11r1 is a area. "An area (11) in r1".
the printed name of the a11r1 is "Room 1".
Understand "Area 11 in Room 1" as a11r1.
the parent of the a11r1 is r1.
the X of the a11r1 is "545108.751214609".
the Y of the a11r1 is "1609625.2932652037".

[create a12r1]
a12r1 is a area. "An area (12) in r1".
the printed name of the a12r1 is "Room 1".
Understand "Area 12 in Room 1" as a12r1.
the parent of the a12r1 is r1.
the X of the a12r1 is "546768.0".
the Y of the a12r1 is "1609809.0".

[create a13r1]
a13r1 is a area. "An area (13) in r1".
the printed name of the a13r1 is "Room 1".
Understand "Area 13 in Room 1" as a13r1.
the parent of the a13r1 is r1.
the X of the a13r1 is "546794.784532571".
the Y of the a13r1 is "1609793.6450085994".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "546485.0".
the Y of the a14r1 is "1609803.0".

[create a15r1]
a15r1 is a area. "An area (15) in r1".
the printed name of the a15r1 is "Room 1".
Understand "Area 15 in Room 1" as a15r1.
the parent of the a15r1 is r1.
the X of the a15r1 is "546563.4811341946".
the Y of the a15r1 is "1609765.6666666667".

[create a34r1]
a34r1 is a area. "An area (34) in r1".
the printed name of the a34r1 is "Room 1".
Understand "Area 34 in Room 1" as a34r1.
the parent of the a34r1 is r1.
the X of the a34r1 is "546485.0".
the Y of the a34r1 is "1609290.0".

[create a35r1]
a35r1 is a area. "An area (35) in r1".
the printed name of the a35r1 is "Room 1".
Understand "Area 35 in Room 1" as a35r1.
the parent of the a35r1 is r1.
the X of the a35r1 is "546548.7508833924".
the Y of the a35r1 is "1609326.125".

[create a58r1]
a58r1 is a area. "An area (58) in r1".
the printed name of the a58r1 is "Room 1".
Understand "Area 58 in Room 1" as a58r1.
the parent of the a58r1 is r1.
the X of the a58r1 is "547708.0".
the Y of the a58r1 is "1610104.0".

[create a59r1]
a59r1 is a area. "An area (59) in r1".
the printed name of the a59r1 is "Room 1".
Understand "Area 59 in Room 1" as a59r1.
the parent of the a59r1 is r1.
the X of the a59r1 is "547687.9999999998".
the Y of the a59r1 is "1610053.7935430459".

[create a60r1]
a60r1 is a area. "An area (60) in r1".
the printed name of the a60r1 is "Room 1".
Understand "Area 60 in Room 1" as a60r1.
the parent of the a60r1 is r1.
the X of the a60r1 is "545071.0".
the Y of the a60r1 is "1610077.0".

[create a61r1]
a61r1 is a area. "An area (61) in r1".
the printed name of the a61r1 is "Room 1".
Understand "Area 61 in Room 1" as a61r1.
the parent of the a61r1 is r1.
the X of the a61r1 is "545100.944940476".
the Y of the a61r1 is "1610100.0".

[create a63r1]
a63r1 is a area. "An area (63) in r1".
the printed name of the a63r1 is "Room 1".
Understand "Area 63 in Room 1" as a63r1.
the parent of the a63r1 is r1.
the X of the a63r1 is "546877.1557637102".
the Y of the a63r1 is "1610039.4674371534".

[create a64r1]
a64r1 is a area. "An area (64) in r1".
the printed name of the a64r1 is "Room 1".
Understand "Area 64 in Room 1" as a64r1.
the parent of the a64r1 is r1.
the X of the a64r1 is "546887.9063773802".
the Y of the a64r1 is "1609658.3939594275".

[create a65r1]
a65r1 is a area. "An area (65) in r1".
the printed name of the a65r1 is "Room 1".
Understand "Area 65 in Room 1" as a65r1.
the parent of the a65r1 is r1.
the X of the a65r1 is "545084.7475343698".
the Y of the a65r1 is "1609658.466890702".

[create a66r1]
a66r1 is a area. "An area (66) in r1".
the printed name of the a66r1 is "Room 1".
Understand "Area 66 in Room 1" as a66r1.
the parent of the a66r1 is r1.
the X of the a66r1 is "546843.8120391241".
the Y of the a66r1 is "1609809.014742796".

[create a67r1]
a67r1 is a area. "An area (67) in r1".
the printed name of the a67r1 is "Room 1".
Understand "Area 67 in Room 1" as a67r1.
the parent of the a67r1 is r1.
the X of the a67r1 is "546626.2935236096".
the Y of the a67r1 is "1609741.1378269286".

[create a68r1]
a68r1 is a area. "An area (68) in r1".
the printed name of the a68r1 is "Room 1".
Understand "Area 68 in Room 1" as a68r1.
the parent of the a68r1 is r1.
the X of the a68r1 is "546762.6954787462".
the Y of the a68r1 is "1609963.0669923371".

[create a69r1]
a69r1 is a area. "An area (69) in r1".
the printed name of the a69r1 is "Room 1".
Understand "Area 69 in Room 1" as a69r1.
the parent of the a69r1 is r1.
the X of the a69r1 is "546762.7118447954".
the Y of the a69r1 is "1609968.508263056".

[create a76r1]
a76r1 is a area. "An area (76) in r1".
the printed name of the a76r1 is "Room 1".
Understand "Area 76 in Room 1" as a76r1.
the parent of the a76r1 is r1.
the X of the a76r1 is "546647.3590770201".
the Y of the a76r1 is "1609377.3054104478".

[create a85r1]
a85r1 is a area. "An area (85) in r1".
the printed name of the a85r1 is "Room 1".
Understand "Area 85 in Room 1" as a85r1.
the parent of the a85r1 is r1.
the X of the a85r1 is "547614.3745890856".
the Y of the a85r1 is "1609810.01862569".

[create a88r1]
a88r1 is a area. "An area (88) in r1".
the printed name of the a88r1 is "Room 1".
Understand "Area 88 in Room 1" as a88r1.
the parent of the a88r1 is r1.
the X of the a88r1 is "547590.7771323381".
the Y of the a88r1 is "1609927.7466226632".

[create a89r1]
a89r1 is a area. "An area (89) in r1".
the printed name of the a89r1 is "Room 1".
Understand "Area 89 in Room 1" as a89r1.
the parent of the a89r1 is r1.
the X of the a89r1 is "545194.3733049509".
the Y of the a89r1 is "1609986.653545771".

[create a90r1]
a90r1 is a area. "An area (90) in r1".
the printed name of the a90r1 is "Room 1".
Understand "Area 90 in Room 1" as a90r1.
the parent of the a90r1 is r1.
the X of the a90r1 is "545983.0".
the Y of the a90r1 is "1609940.1754812326".

[create a38r2]
a38r2 is a area. "An area (38) in r2".
the printed name of the a38r2 is "Room 2".
Understand "Area 38 in Room 2" as a38r2.
the parent of the a38r2 is r2.
the X of the a38r2 is "545556.0".
the Y of the a38r2 is "1607991.0".

[create a39r2]
a39r2 is a area. "An area (39) in r2".
the printed name of the a39r2 is "Room 2".
Understand "Area 39 in Room 2" as a39r2.
the parent of the a39r2 is r2.
the X of the a39r2 is "545587.2137161085".
the Y of the a39r2 is "1608016.633971292".

[create a42r2]
a42r2 is a area. "An area (42) in r2".
the printed name of the a42r2 is "Room 2".
Understand "Area 42 in Room 2" as a42r2.
the parent of the a42r2 is r2.
the X of the a42r2 is "545547.0".
the Y of the a42r2 is "1608080.0".

[create a43r2]
a43r2 is a area. "An area (43) in r2".
the printed name of the a43r2 is "Room 2".
Understand "Area 43 in Room 2" as a43r2.
the parent of the a43r2 is r2.
the X of the a43r2 is "545577.8730024241".
the Y of the a43r2 is "1608060.644460919".

[create a48r2]
a48r2 is a area. "An area (48) in r2".
the printed name of the a48r2 is "Room 2".
Understand "Area 48 in Room 2" as a48r2.
the parent of the a48r2 is r2.
the X of the a48r2 is "546398.0".
the Y of the a48r2 is "1607991.0".

[create a49r2]
a49r2 is a area. "An area (49) in r2".
the printed name of the a49r2 is "Room 2".
Understand "Area 49 in Room 2" as a49r2.
the parent of the a49r2 is r2.
the X of the a49r2 is "546366.7862838916".
the Y of the a49r2 is "1608016.633971292".

[create a50r2]
a50r2 is a area. "An area (50) in r2".
the printed name of the a50r2 is "Room 2".
Understand "Area 50 in Room 2" as a50r2.
the parent of the a50r2 is r2.
the X of the a50r2 is "546407.0".
the Y of the a50r2 is "1608080.0".

[create a51r2]
a51r2 is a area. "An area (51) in r2".
the printed name of the a51r2 is "Room 2".
Understand "Area 51 in Room 2" as a51r2.
the parent of the a51r2 is r2.
the X of the a51r2 is "546363.9383292025".
the Y of the a51r2 is "1608061.8770228897".

[create a77r2]
a77r2 is a area. "An area (77) in r2".
the printed name of the a77r2 is "Room 2".
Understand "Area 77 in Room 2" as a77r2.
the parent of the a77r2 is r2.
the X of the a77r2 is "545601.3945333334".
the Y of the a77r2 is "1608040.2686666665".

[create a81r2]
a81r2 is a area. "An area (81) in r2".
the printed name of the a81r2 is "Room 2".
Understand "Area 81 in Room 2" as a81r2.
the parent of the a81r2 is r2.
the X of the a81r2 is "546349.7290448344".
the Y of the a81r2 is "1608045.062703054".

[create a87r2]
a87r2 is a area. "An area (87) in r2".
the printed name of the a87r2 is "Room 2".
Understand "Area 87 in Room 2" as a87r2.
the parent of the a87r2 is r2.
the X of the a87r2 is "545983.0".
the Y of the a87r2 is "1608168.577830189".

[create a24r3]
a24r3 is a area. "An area (24) in r3".
the printed name of the a24r3 is "Room 3".
Understand "Area 24 in Room 3" as a24r3.
the parent of the a24r3 is r3.
the X of the a24r3 is "548491.0".
the Y of the a24r3 is "1608003.0".

[create a25r3]
a25r3 is a area. "An area (25) in r3".
the printed name of the a25r3 is "Room 3".
Understand "Area 25 in Room 3" as a25r3.
the parent of the a25r3 is r3.
the X of the a25r3 is "548447.8503889913".
the Y of the a25r3 is "1608061.5027281865".

[create a52r3]
a52r3 is a area. "An area (52) in r3".
the printed name of the a52r3 is "Room 3".
Understand "Area 52 in Room 3" as a52r3.
the parent of the a52r3 is r3.
the X of the a52r3 is "548301.0".
the Y of the a52r3 is "1607969.0".

[create a53r3]
a53r3 is a area. "An area (53) in r3".
the printed name of the a53r3 is "Room 3".
Understand "Area 53 in Room 3" as a53r3.
the parent of the a53r3 is r3.
the X of the a53r3 is "548324.5690527358".
the Y of the a53r3 is "1608019.9180386337".

[create a71r3]
a71r3 is a area. "An area (71) in r3".
the printed name of the a71r3 is "Room 3".
Understand "Area 71 in Room 3" as a71r3.
the parent of the a71r3 is r3.
the X of the a71r3 is "548387.0164376835".
the Y of the a71r3 is "1608130.1614343065".

[create a72r3]
a72r3 is a area. "An area (72) in r3".
the printed name of the a72r3 is "Room 3".
Understand "Area 72 in Room 3" as a72r3.
the parent of the a72r3 is r3.
the X of the a72r3 is "548389.7654538173".
the Y of the a72r3 is "1608149.5884375954".

[create a16r4]
a16r4 is a area. "An area (16) in r4".
the printed name of the a16r4 is "Room 4".
Understand "Area 16 in Room 4" as a16r4.
the parent of the a16r4 is r4.
the X of the a16r4 is "549723.0".
the Y of the a16r4 is "1610024.0".

[create a17r4]
a17r4 is a area. "An area (17) in r4".
the printed name of the a17r4 is "Room 4".
Understand "Area 17 in Room 4" as a17r4.
the parent of the a17r4 is r4.
the X of the a17r4 is "549680.94".
the Y of the a17r4 is "1610042.5189999999".

[create a18r4]
a18r4 is a area. "An area (18) in r4".
the printed name of the a18r4 is "Room 4".
Understand "Area 18 in Room 4" as a18r4.
the parent of the a18r4 is r4.
the X of the a18r4 is "549714.0".
the Y of the a18r4 is "1610114.0".

[create a19r4]
a19r4 is a area. "An area (19) in r4".
the printed name of the a19r4 is "Room 4".
Understand "Area 19 in Room 4" as a19r4.
the parent of the a19r4 is r4.
the X of the a19r4 is "549682.9123931624".
the Y of the a19r4 is "1610088.1662393163".

[create a56r4]
a56r4 is a area. "An area (56) in r4".
the printed name of the a56r4 is "Room 4".
Understand "Area 56 in Room 4" as a56r4.
the parent of the a56r4 is r4.
the X of the a56r4 is "549170.0".
the Y of the a56r4 is "1610104.0".

[create a57r4]
a57r4 is a area. "An area (57) in r4".
the printed name of the a57r4 is "Room 4".
Understand "Area 57 in Room 4" as a57r4.
the parent of the a57r4 is r4.
the X of the a57r4 is "549205.0".
the Y of the a57r4 is "1610047.742218543".

[create a62r4]
a62r4 is a area. "An area (62) in r4".
the printed name of the a62r4 is "Room 4".
Understand "Area 62 in Room 4" as a62r4.
the parent of the a62r4 is r4.
the X of the a62r4 is "549665.8606168447".
the Y of the a62r4 is "1610059.8602906289".

[create a84r4]
a84r4 is a area. "An area (84) in r4".
the printed name of the a84r4 is "Room 4".
Understand "Area 84 in Room 4" as a84r4.
the parent of the a84r4 is r4.
the X of the a84r4 is "549260.9001965043".
the Y of the a84r4 is "1609793.9469272906".

[create a91r4]
a91r4 is a area. "An area (91) in r4".
the printed name of the a91r4 is "Room 4".
Understand "Area 91 in Room 4" as a91r4.
the parent of the a91r4 is r4.
the X of the a91r4 is "549289.6187909034".
the Y of the a91r4 is "1609923.5461257007".

[create a20r5]
a20r5 is a area. "An area (20) in r5".
the printed name of the a20r5 is "Room 5".
Understand "Area 20 in Room 5" as a20r5.
the parent of the a20r5 is r5.
the X of the a20r5 is "549791.0".
the Y of the a20r5 is "1608080.0".

[create a21r5]
a21r5 is a area. "An area (21) in r5".
the printed name of the a21r5 is "Room 5".
Understand "Area 21 in Room 5" as a21r5.
the parent of the a21r5 is r5.
the X of the a21r5 is "549748.0247108882".
the Y of the a21r5 is "1608062.2977852935".

[create a36r5]
a36r5 is a area. "An area (36) in r5".
the printed name of the a36r5 is "Room 5".
Understand "Area 36 in Room 5" as a36r5.
the parent of the a36r5 is r5.
the X of the a36r5 is "548940.0".
the Y of the a36r5 is "1607991.0".

[create a37r5]
a37r5 is a area. "An area (37) in r5".
the printed name of the a37r5 is "Room 5".
Understand "Area 37 in Room 5" as a37r5.
the parent of the a37r5 is r5.
the X of the a37r5 is "548971.0234020579".
the Y of the a37r5 is "1608017.0166743884".

[create a44r5]
a44r5 is a area. "An area (44) in r5".
the printed name of the a44r5 is "Room 5".
Understand "Area 44 in Room 5" as a44r5.
the parent of the a44r5 is r5.
the X of the a44r5 is "548930.0".
the Y of the a44r5 is "1608080.0".

[create a45r5]
a45r5 is a area. "An area (45) in r5".
the printed name of the a45r5 is "Room 5".
Understand "Area 45 in Room 5" as a45r5.
the parent of the a45r5 is r5.
the X of the a45r5 is "548961.3505784648".
the Y of the a45r5 is "1608060.9916380297".

[create a46r5]
a46r5 is a area. "An area (46) in r5".
the printed name of the a46r5 is "Room 5".
Understand "Area 46 in Room 5" as a46r5.
the parent of the a46r5 is r5.
the X of the a46r5 is "549781.0".
the Y of the a46r5 is "1607991.0".

[create a47r5]
a47r5 is a area. "An area (47) in r5".
the printed name of the a47r5 is "Room 5".
Understand "Area 47 in Room 5" as a47r5.
the parent of the a47r5 is r5.
the X of the a47r5 is "549749.9765979422".
the Y of the a47r5 is "1608017.0166743887".

[create a79r5]
a79r5 is a area. "An area (79) in r5".
the printed name of the a79r5 is "Room 5".
Understand "Area 79 in Room 5" as a79r5.
the parent of the a79r5 is r5.
the X of the a79r5 is "548984.9132133439".
the Y of the a79r5 is "1608040.2221422398".

[create a80r5]
a80r5 is a area. "An area (80) in r5".
the printed name of the a80r5 is "Room 5".
Understand "Area 80 in Room 5" as a80r5.
the parent of the a80r5 is r5.
the X of the a80r5 is "549733.2584382974".
the Y of the a80r5 is "1608044.9474149996".

[create a86r5]
a86r5 is a area. "An area (86) in r5".
the printed name of the a86r5 is "Room 5".
Understand "Area 86 in Room 5" as a86r5.
the parent of the a86r5 is r5.
the X of the a86r5 is "549370.0668298497".
the Y of the a86r5 is "1608165.8958366967".

[create d0]
d0 is a door. "A door between a72r3 and a84r4".
d0 is northeast of a72r3 and southwest of a84r4.

[create d2]
d2 is a door. "A door between a85r1 and a72r3".
d2 is southeast of a85r1 and northwest of a72r3.

[create d4]
d4 is a door. "A door between a84r4 and a86r5".
d4 is south of a84r4 and north of a86r5.

[create d6]
d6 is a door. "A door between a90r1 and a87r2".
d6 is south of a90r1 and north of a87r2.

[create d8]
d8 is a door. "A door between a92r0 and a76r1".
d8 is northwest of a92r0 and southeast of a76r1.

[create d9]
d9 is a door. "A door between a92r0 and a85r1".
d9 is north of a92r0 and south of a85r1.

northwest of a0r0 is southeast of a1r0.

southeast of a2r1 is northwest of a3r1.

north of a4r1 is south of a5r1.

northwest of a6r1 is southeast of a7r1.

northeast of a8r1 is southwest of a9r1.

north of a10r1 is south of a11r1.

southeast of a12r1 is northwest of a13r1.

southeast of a14r1 is northwest of a15r1.

northwest of a16r4 is southeast of a17r4.

southwest of a18r4 is northeast of a19r4.

west of a20r5 is east of a21r5.

northwest of a22r0 is southeast of a23r0.

northwest of a24r3 is southeast of a25r3.

southeast of a26r0 is northwest of a27r0.

south of a28r0 is north of a29r0.

northwest of a30r0 is southeast of a31r0.

southwest of a32r0 is northeast of a33r0.

northeast of a34r1 is southwest of a35r1.

northeast of a36r5 is southwest of a37r5.

northeast of a38r2 is southwest of a39r2.

southwest of a40r0 is northeast of a41r0.

southeast of a42r2 is northwest of a43r2.

southeast of a44r5 is northwest of a45r5.

northwest of a46r5 is southeast of a47r5.

northwest of a48r2 is southeast of a49r2.

southwest of a50r2 is northeast of a51r2.

northeast of a52r3 is southwest of a53r3.

northeast of a54r0 is southwest of a55r0.

southeast of a56r4 is northwest of a57r4.

south of a58r1 is north of a59r1.

northeast of a60r1 is southwest of a61r1.

southwest of a19r4 is northeast of a62r4.

south of a3r1 is north of a63r1.

northeast of a5r1 is southwest of a64r1.

west of a7r1 is east of a64r1.

east of a9r1 is west of a65r1.

northwest of a11r1 is southeast of a65r1.

east of a13r1 is west of a66r1.

east of a15r1 is west of a67r1.

north of a68r1 is south of a69r1.

northwest of a17r4 is southeast of a62r4.

north of a23r0 is south of a70r0.

northwest of a25r3 is southeast of a71r3.

north of a71r3 is south of a72r3.

southeast of a27r0 is northwest of a73r0.

southwest of a29r0 is northeast of a73r0.

northwest of a31r0 is southeast of a74r0.

southwest of a33r0 is northeast of a75r0.

northeast of a35r1 is southwest of a76r1.

northeast of a39r2 is southwest of a77r2.

southwest of a41r0 is northeast of a78r0.

southeast of a43r2 is northwest of a77r2.

southeast of a45r5 is northwest of a79r5.

northwest of a47r5 is southeast of a80r5.

northwest of a49r2 is southeast of a81r2.

southwest of a51r2 is northeast of a81r2.

southwest of a21r5 is northeast of a80r5.

northwest of a1r0 is southeast of a78r0.

northeast of a37r5 is southwest of a79r5.

west of a70r0 is east of a82r0.

northeast of a53r3 is southwest of a71r3.

northeast of a70r0 is southwest of a83r0.

northeast of a55r0 is southwest of a82r0.

south of a73r0 is north of a82r0.

west of a78r0 is east of a83r0.

west of a81r2 is east of a87r2.

west of a80r5 is east of a86r5.

east of a77r2 is west of a87r2.

east of a79r5 is west of a86r5.

southwest of a59r1 is northeast of a88r1.

east of a89r1 is west of a90r1.

west of a69r1 is east of a90r1.

southwest of a63r1 is northeast of a69r1.

southeast of a61r1 is northwest of a89r1.

north of a65r1 is south of a89r1.

northeast of a67r1 is southwest of a68r1.

northwest of a66r1 is southeast of a68r1.

north of a84r4 is south of a91r4.

north of a85r1 is south of a88r1.

west of a62r4 is east of a91r4.

southeast of a57r4 is northwest of a91r4.

east of a63r1 is west of a88r1.

southwest of a74r0 is northeast of a83r0.

north of a74r0 is south of a75r0.

northwest of a75r0 is southeast of a92r0.

south of a67r1 is north of a76r1.

north of a64r1 is south of a66r1.

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
the player is in a12r1.

the orientation of the player is 6.
