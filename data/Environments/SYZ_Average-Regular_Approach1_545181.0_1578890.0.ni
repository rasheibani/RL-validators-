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

[create a5r0]
a5r0 is a area. "An area (5) in r0".
the printed name of the a5r0 is "Room 0".
Understand "Area 5 in Room 0" as a5r0.
the parent of the a5r0 is r0.
the X of the a5r0 is "545568.7588065893".
the Y of the a5r0 is "1579638.1469224335".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "546150.3192205713".
the Y of the a6r0 is "1579532.7984514583".

[create a14r0]
a14r0 is a area. "An area (14) in r0".
the printed name of the a14r0 is "Room 0".
Understand "Area 14 in Room 0" as a14r0.
the parent of the a14r0 is r0.
the X of the a14r0 is "545322.2765708305".
the Y of the a14r0 is "1579341.3817745445".

[create a15r0]
a15r0 is a area. "An area (15) in r0".
the printed name of the a15r0 is "Room 0".
Understand "Area 15 in Room 0" as a15r0.
the parent of the a15r0 is r0.
the X of the a15r0 is "545343.35907702".
the Y of the a15r0 is "1578977.3054104475".

[create a18r0]
a18r0 is a area. "An area (18) in r0".
the printed name of the a18r0 is "Room 0".
Understand "Area 18 in Room 0" as a18r0.
the parent of the a18r0 is r0.
the X of the a18r0 is "546520.6041135422".
the Y of the a18r0 is "1579662.7544951467".

[create a25r0]
a25r0 is a area. "An area (25) in r0".
the printed name of the a25r0 is "Room 0".
Understand "Area 25 in Room 0" as a25r0.
the parent of the a25r0 is r0.
the X of the a25r0 is "545181.0".
the Y of the a25r0 is "1578890.0".

[create a26r0]
a26r0 is a area. "An area (26) in r0".
the printed name of the a26r0 is "Room 0".
Understand "Area 26 in Room 0" as a26r0.
the parent of the a26r0 is r0.
the X of the a26r0 is "545244.7508833923".
the Y of the a26r0 is "1578926.125".

[create a31r0]
a31r0 is a area. "An area (31) in r0".
the printed name of the a31r0 is "Room 0".
Understand "Area 31 in Room 0" as a31r0.
the parent of the a31r0 is r0.
the X of the a31r0 is "545181.0".
the Y of the a31r0 is "1579403.0".

[create a32r0]
a32r0 is a area. "An area (32) in r0".
the printed name of the a32r0 is "Room 0".
Understand "Area 32 in Room 0" as a32r0.
the parent of the a32r0 is r0.
the X of the a32r0 is "545263.1833333333".
the Y of the a32r0 is "1579365.6666666667".

[create a57r0]
a57r0 is a area. "An area (57) in r0".
the printed name of the a57r0 is "Room 0".
Understand "Area 57 in Room 0" as a57r0.
the parent of the a57r0 is r0.
the X of the a57r0 is "546576.0".
the Y of the a57r0 is "1579624.0".

[create a58r0]
a58r0 is a area. "An area (58) in r0".
the printed name of the a58r0 is "Room 0".
Understand "Area 58 in Room 0" as a58r0.
the parent of the a58r0 is r0.
the X of the a58r0 is "546539.7408262398".
the Y of the a58r0 is "1579643.0990826238".

[create a63r0]
a63r0 is a area. "An area (63) in r0".
the printed name of the a63r0 is "Room 0".
Understand "Area 63 in Room 0" as a63r0.
the parent of the a63r0 is r0.
the X of the a63r0 is "546567.0".
the Y of the a63r0 is "1579714.0".

[create a64r0]
a64r0 is a area. "An area (64) in r0".
the printed name of the a64r0 is "Room 0".
Understand "Area 64 in Room 0" as a64r0.
the parent of the a64r0 is r0.
the X of the a64r0 is "546535.9123931623".
the Y of the a64r0 is "1579688.166239316".

[create a65r0]
a65r0 is a area. "An area (65) in r0".
the printed name of the a65r0 is "Room 0".
Understand "Area 65 in Room 0" as a65r0.
the parent of the a65r0 is r0.
the X of the a65r0 is "545531.0".
the Y of the a65r0 is "1579732.0".

[create a66r0]
a66r0 is a area. "An area (66) in r0".
the printed name of the a66r0 is "Room 0".
Understand "Area 66 in Room 0" as a66r0.
the parent of the a66r0 is r0.
the X of the a66r0 is "545566.375".
the Y of the a66r0 is "1579644.2712765958".

[create a3r1]
a3r1 is a area. "An area (3) in r1".
the printed name of the a3r1 is "Room 1".
Understand "Area 3 in Room 1" as a3r1.
the parent of the a3r1 is r1.
the X of the a3r1 is "546062.7882951223".
the Y of the a3r1 is "1577682.7079821012".

[create a4r1]
a4r1 is a area. "An area (4) in r1".
the printed name of the a4r1 is "Room 1".
Understand "Area 4 in Room 1" as a4r1.
the parent of the a4r1 is r1.
the X of the a4r1 is "546314.5701081634".
the Y of the a4r1 is "1577979.9940556635".

[create a19r1]
a19r1 is a area. "An area (19) in r1".
the printed name of the a19r1 is "Room 1".
Understand "Area 19 in Room 1" as a19r1.
the parent of the a19r1 is r1.
the X of the a19r1 is "545197.0".
the Y of the a19r1 is "1578131.0".

[create a20r1]
a20r1 is a area. "An area (20) in r1".
the printed name of the a20r1 is "Room 1".
Understand "Area 20 in Room 1" as a20r1.
the parent of the a20r1 is r1.
the X of the a20r1 is "545236.6586345381".
the Y of the a20r1 is "1578097.8333333333".

[create a21r1]
a21r1 is a area. "An area (21) in r1".
the printed name of the a21r1 is "Room 1".
Understand "Area 21 in Room 1" as a21r1.
the parent of the a21r1 is r1.
the X of the a21r1 is "545280.0".
the Y of the a21r1 is "1578168.0".

[create a22r1]
a22r1 is a area. "An area (22) in r1".
the printed name of the a22r1 is "Room 1".
Understand "Area 22 in Room 1" as a22r1.
the parent of the a22r1 is r1.
the X of the a22r1 is "545270.4164161286".
the Y of the a22r1 is "1578133.7010124677".

[create a23r1]
a23r1 is a area. "An area (23) in r1".
the printed name of the a23r1 is "Room 1".
Understand "Area 23 in Room 1" as a23r1.
the parent of the a23r1 is r1.
the X of the a23r1 is "546475.0".
the Y of the a23r1 is "1578426.0".

[create a24r1]
a24r1 is a area. "An area (24) in r1".
the printed name of the a24r1 is "Room 1".
Understand "Area 24 in Room 1" as a24r1.
the parent of the a24r1 is r1.
the X of the a24r1 is "546406.0821093749".
the Y of the a24r1 is "1578390.375".

[create a41r1]
a41r1 is a area. "An area (41) in r1".
the printed name of the a41r1 is "Room 1".
Understand "Area 41 in Room 1" as a41r1.
the parent of the a41r1 is r1.
the X of the a41r1 is "546109.0".
the Y of the a41r1 is "1577569.0".

[create a42r1]
a42r1 is a area. "An area (42) in r1".
the printed name of the a42r1 is "Room 1".
Understand "Area 42 in Room 1" as a42r1.
the parent of the a42r1 is r1.
the X of the a42r1 is "546070.1111111111".
the Y of the a42r1 is "1577663.1566122046".

[create a51r1]
a51r1 is a area. "An area (51) in r1".
the printed name of the a51r1 is "Room 1".
Understand "Area 51 in Room 1" as a51r1.
the parent of the a51r1 is r1.
the X of the a51r1 is "546475.0".
the Y of the a51r1 is "1577910.0".

[create a52r1]
a52r1 is a area. "An area (52) in r1".
the printed name of the a52r1 is "Room 1".
Understand "Area 52 in Room 1" as a52r1.
the parent of the a52r1 is r1.
the X of the a52r1 is "546390.2935206869".
the Y of the a52r1 is "1577948.5".

[create a72r1]
a72r1 is a area. "An area (72) in r1".
the printed name of the a72r1 is "Room 1".
Understand "Area 72 in Room 1" as a72r1.
the parent of the a72r1 is r1.
the X of the a72r1 is "545147.0".
the Y of the a72r1 is "1577680.0".

[create a73r1]
a73r1 is a area. "An area (73) in r1".
the printed name of the a73r1 is "Room 1".
Understand "Area 73 in Room 1" as a73r1.
the parent of the a73r1 is r1.
the X of the a73r1 is "545195.4282420991".
the Y of the a73r1 is "1577703.1313011711".

[create a74r1]
a74r1 is a area. "An area (74) in r1".
the printed name of the a74r1 is "Room 1".
Understand "Area 74 in Room 1" as a74r1.
the parent of the a74r1 is r1.
the X of the a74r1 is "545239.2405909916".
the Y of the a74r1 is "1578096.5702544395".

[create a75r1]
a75r1 is a area. "An area (75) in r1".
the printed name of the a75r1 is "Room 1".
Understand "Area 75 in Room 1" as a75r1.
the parent of the a75r1 is r1.
the X of the a75r1 is "546305.798041391".
the Y of the a75r1 is "1578333.6020645646".

[create a80r1]
a80r1 is a area. "An area (80) in r1".
the printed name of the a80r1 is "Room 1".
Understand "Area 80 in Room 1" as a80r1.
the parent of the a80r1 is r1.
the X of the a80r1 is "545297.9453007281".
the Y of the a80r1 is "1577757.851663235".

[create a12r2]
a12r2 is a area. "An area (12) in r2".
the printed name of the a12r2 is "Room 2".
Understand "Area 12 in Room 2" as a12r2.
the parent of the a12r2 is r2.
the X of the a12r2 is "546706.1730804583".
the Y of the a12r2 is "1578459.7005024815".

[create a37r2]
a37r2 is a area. "An area (37) in r2".
the printed name of the a37r2 is "Room 2".
Understand "Area 37 in Room 2" as a37r2.
the parent of the a37r2 is r2.
the X of the a37r2 is "547126.0".
the Y of the a37r2 is "1577591.0".

[create a38r2]
a38r2 is a area. "An area (38) in r2".
the printed name of the a38r2 is "Room 2".
Understand "Area 38 in Room 2" as a38r2.
the parent of the a38r2 is r2.
the X of the a38r2 is "547094.7862838916".
the Y of the a38r2 is "1577616.633971292".

[create a39r2]
a39r2 is a area. "An area (39) in r2".
the printed name of the a39r2 is "Room 2".
Understand "Area 39 in Room 2" as a39r2.
the parent of the a39r2 is r2.
the X of the a39r2 is "547135.0".
the Y of the a39r2 is "1577680.0".

[create a40r2]
a40r2 is a area. "An area (40) in r2".
the printed name of the a40r2 is "Room 2".
Understand "Area 40 in Room 2" as a40r2.
the parent of the a40r2 is r2.
the X of the a40r2 is "547091.9383292024".
the Y of the a40r2 is "1577661.8770228894".

[create a53r2]
a53r2 is a area. "An area (53) in r2".
the printed name of the a53r2 is "Room 2".
Understand "Area 53 in Room 2" as a53r2.
the parent of the a53r2 is r2.
the X of the a53r2 is "546284.0".
the Y of the a53r2 is "1577591.0".

[create a54r2]
a54r2 is a area. "An area (54) in r2".
the printed name of the a54r2 is "Room 2".
Understand "Area 54 in Room 2" as a54r2.
the parent of the a54r2 is r2.
the X of the a54r2 is "546315.2137161085".
the Y of the a54r2 is "1577616.633971292".

[create a55r2]
a55r2 is a area. "An area (55) in r2".
the printed name of the a55r2 is "Room 2".
Understand "Area 55 in Room 2" as a55r2.
the parent of the a55r2 is r2.
the X of the a55r2 is "546275.0".
the Y of the a55r2 is "1577680.0".

[create a56r2]
a56r2 is a area. "An area (56) in r2".
the printed name of the a56r2 is "Room 2".
Understand "Area 56 in Room 2" as a56r2.
the parent of the a56r2 is r2.
the X of the a56r2 is "546305.8730024243".
the Y of the a56r2 is "1577660.6444609193".

[create a76r2]
a76r2 is a area. "An area (76) in r2".
the printed name of the a76r2 is "Room 2".
Understand "Area 76 in Room 2" as a76r2.
the parent of the a76r2 is r2.
the X of the a76r2 is "547077.7290448343".
the Y of the a76r2 is "1577645.062703054".

[create a79r2]
a79r2 is a area. "An area (79) in r2".
the printed name of the a79r2 is "Room 2".
Understand "Area 79 in Room 2" as a79r2.
the parent of the a79r2 is r2.
the X of the a79r2 is "546329.3945333333".
the Y of the a79r2 is "1577640.2686666667".

[create a81r2]
a81r2 is a area. "An area (81) in r2".
the printed name of the a81r2 is "Room 2".
Understand "Area 81 in Room 2" as a81r2.
the parent of the a81r2 is r2.
the X of the a81r2 is "546711.0".
the Y of the a81r2 is "1577768.577830189".

[create a7r3]
a7r3 is a area. "An area (7) in r3".
the printed name of the a7r3 is "Room 3".
Understand "Area 7 in Room 3" as a7r3.
the parent of the a7r3 is r3.
the X of the a7r3 is "547664.4262276839".
the Y of the a7r3 is "1579607.9647864522".

[create a8r3]
a8r3 is a area. "An area (8) in r3".
the printed name of the a8r3 is "Room 3".
Understand "Area 8 in Room 3" as a8r3.
the parent of the a8r3 is r3.
the X of the a8r3 is "548747.4980734903".
the Y of the a8r3 is "1579557.3206812765".

[create a9r3]
a9r3 is a area. "An area (9) in r3".
the printed name of the a9r3 is "Room 3".
Understand "Area 9 in Room 3" as a9r3.
the parent of the a9r3 is r3.
the X of the a9r3 is "546997.7072666292".
the Y of the a9r3 is "1579661.4777749956".

[create a10r3]
a10r3 is a area. "An area (10) in r3".
the printed name of the a10r3 is "Room 3".
Understand "Area 10 in Room 3" as a10r3.
the parent of the a10r3 is r3.
the X of the a10r3 is "547359.6800897496".
the Y of the a10r3 is "1579553.8437647612".

[create a11r3]
a11r3 is a area. "An area (11) in r3".
the printed name of the a11r3 is "Room 3".
Understand "Area 11 in Room 3" as a11r3.
the parent of the a11r3 is r3.
the X of the a11r3 is "547401.7200386736".
the Y of the a11r3 is "1579574.374902608".

[create a13r3]
a13r3 is a area. "An area (13) in r3".
the printed name of the a13r3 is "Room 3".
Understand "Area 13 in Room 3" as a13r3.
the parent of the a13r3 is r3.
the X of the a13r3 is "547297.0157834981".
the Y of the a13r3 is "1579368.6403156356".

[create a16r3]
a16r3 is a area. "An area (16) in r3".
the printed name of the a16r3 is "Room 3".
Understand "Area 16 in Room 3" as a16r3.
the parent of the a16r3 is r3.
the X of the a16r3 is "547350.0583993593".
the Y of the a16r3 is "1579263.1800747695".

[create a17r3]
a17r3 is a area. "An area (17) in r3".
the printed name of the a17r3 is "Room 3".
Understand "Area 17 in Room 3" as a17r3.
the parent of the a17r3 is r3.
the X of the a17r3 is "548856.1617869516".
the Y of the a17r3 is "1579609.4858537384".

[create a27r3]
a27r3 is a area. "An area (27) in r3".
the printed name of the a27r3 is "Room 3".
Understand "Area 27 in Room 3" as a27r3.
the parent of the a27r3 is r3.
the X of the a27r3 is "547303.0".
the Y of the a27r3 is "1579225.0".

[create a28r3]
a28r3 is a area. "An area (28) in r3".
the printed name of the a28r3 is "Room 3".
Understand "Area 28 in Room 3" as a28r3.
the parent of the a28r3 is r3.
the X of the a28r3 is "547327.0920404801".
the Y of the a28r3 is "1579223.2470097253".

[create a29r3]
a29r3 is a area. "An area (29) in r3".
the printed name of the a29r3 is "Room 3".
Understand "Area 29 in Room 3" as a29r3.
the parent of the a29r3 is r3.
the X of the a29r3 is "547386.0".
the Y of the a29r3 is "1579188.0".

[create a30r3]
a30r3 is a area. "An area (30) in r3".
the printed name of the a30r3 is "Room 3".
Understand "Area 30 in Room 3" as a30r3.
the parent of the a30r3 is r3.
the X of the a30r3 is "547377.9921842946".
the Y of the a30r3 is "1579225.8338188233".

[create a33r3]
a33r3 is a area. "An area (33) in r3".
the printed name of the a33r3 is "Room 3".
Understand "Area 33 in Room 3" as a33r3.
the parent of the a33r3 is r3.
the X of the a33r3 is "548968.0".
the Y of the a33r3 is "1579603.0".

[create a34r3]
a34r3 is a area. "An area (34) in r3".
the printed name of the a34r3 is "Room 3".
Understand "Area 34 in Room 3" as a34r3.
the parent of the a34r3 is r3.
the X of the a34r3 is "548904.3734208058".
the Y of the a34r3 is "1579601.0894865775".

[create a35r3]
a35r3 is a area. "An area (35) in r3".
the printed name of the a35r3 is "Room 3".
Understand "Area 35 in Room 3" as a35r3.
the parent of the a35r3 is r3.
the X of the a35r3 is "546941.0".
the Y of the a35r3 is "1579624.0".

[create a36r3]
a36r3 is a area. "An area (36) in r3".
the printed name of the a36r3 is "Room 3".
Understand "Area 36 in Room 3" as a36r3.
the parent of the a36r3 is r3.
the X of the a36r3 is "546979.7029286583".
the Y of the a36r3 is "1579642.477452371".

[create a59r3]
a59r3 is a area. "An area (59) in r3".
the printed name of the a59r3 is "Room 3".
Understand "Area 59 in Room 3" as a59r3.
the parent of the a59r3 is r3.
the X of the a59r3 is "548913.0".
the Y of the a59r3 is "1579704.0".

[create a60r3]
a60r3 is a area. "An area (60) in r3".
the printed name of the a60r3 is "Room 3".
Understand "Area 60 in Room 3" as a60r3.
the parent of the a60r3 is r3.
the X of the a60r3 is "548879.1666666666".
the Y of the a60r3 is "1579652.8382838285".

[create a61r3]
a61r3 is a area. "An area (61) in r3".
the printed name of the a61r3 is "Room 3".
Understand "Area 61 in Room 3" as a61r3.
the parent of the a61r3 is r3.
the X of the a61r3 is "546951.0".
the Y of the a61r3 is "1579714.0".

[create a62r3]
a62r3 is a area. "An area (62) in r3".
the printed name of the a62r3 is "Room 3".
Understand "Area 62 in Room 3" as a62r3.
the parent of the a62r3 is r3.
the X of the a62r3 is "546981.8949627937".
the Y of the a62r3 is "1579687.7894485786".

[create a67r3]
a67r3 is a area. "An area (67) in r3".
the printed name of the a67r3 is "Room 3".
Understand "Area 67 in Room 3" as a67r3.
the parent of the a67r3 is r3.
the X of the a67r3 is "547710.1111111111".
the Y of the a67r3 is "1579704.0".

[create a68r3]
a68r3 is a area. "An area (68) in r3".
the printed name of the a68r3 is "Room 3".
Understand "Area 68 in Room 3" as a68r3.
the parent of the a68r3 is r3.
the X of the a68r3 is "547672.3849518171".
the Y of the a68r3 is "1579704.8699329179".

[create a69r3]
a69r3 is a area. "An area (69) in r3".
the printed name of the a69r3 is "Room 3".
Understand "Area 69 in Room 3" as a69r3.
the parent of the a69r3 is r3.
the X of the a69r3 is "547340.0".
the Y of the a69r3 is "1579704.0".

[create a70r3]
a70r3 is a area. "An area (70) in r3".
the printed name of the a70r3 is "Room 3".
Understand "Area 70 in Room 3" as a70r3.
the parent of the a70r3 is r3.
the X of the a70r3 is "547376.3339535533".
the Y of the a70r3 is "1579713.5".

[create a0r4]
a0r4 is a area. "An area (0) in r4".
the printed name of the a0r4 is "Room 4".
Understand "Area 0 in Room 4" as a0r4.
the parent of the a0r4 is r4.
the X of the a0r4 is "547444.2078042733".
the Y of the a0r4 is "1577750.9693418897".

[create a1r4]
a1r4 is a area. "An area (1) in r4".
the printed name of the a1r4 is "Room 4".
Understand "Area 1 in Room 4" as a1r4.
the parent of the a1r4 is r4.
the X of the a1r4 is "548816.1954237347".
the Y of the a1r4 is "1577731.069391028".

[create a2r4]
a2r4 is a area. "An area (2) in r4".
the printed name of the a2r4 is "Room 4".
Understand "Area 2 in Room 4" as a2r4.
the parent of the a2r4 is r4.
the X of the a2r4 is "548903.8095238095".
the Y of the a2r4 is "1577634.3473951335".

[create a43r4]
a43r4 is a area. "An area (43) in r4".
the printed name of the a43r4 is "Room 4".
Understand "Area 43 in Room 4" as a43r4.
the parent of the a43r4 is r4.
the X of the a43r4 is "547223.0".
the Y of the a43r4 is "1577708.0".

[create a44r4]
a44r4 is a area. "An area (44) in r4".
the printed name of the a44r4 is "Room 4".
Understand "Area 44 in Room 4" as a44r4.
the parent of the a44r4 is r4.
the X of the a44r4 is "547284.9763877094".
the Y of the a44r4 is "1577710.7893767636".

[create a45r4]
a45r4 is a area. "An area (45) in r4".
the printed name of the a45r4 is "Room 4".
Understand "Area 45 in Room 4" as a45r4.
the parent of the a45r4 is r4.
the X of the a45r4 is "547297.0".
the Y of the a45r4 is "1577600.0".

[create a46r4]
a46r4 is a area. "An area (46) in r4".
the printed name of the a46r4 is "Room 4".
Understand "Area 46 in Room 4" as a46r4.
the parent of the a46r4 is r4.
the X of the a46r4 is "547336.1904761905".
the Y of the a46r4 is "1577666.528659612".

[create a47r4]
a47r4 is a area. "An area (47) in r4".
the printed name of the a47r4 is "Room 4".
Understand "Area 47 in Room 4" as a47r4.
the parent of the a47r4 is r4.
the X of the a47r4 is "549017.0".
the Y of the a47r4 is "1578088.0".

[create a48r4]
a48r4 is a area. "An area (48) in r4".
the printed name of the a48r4 is "Room 4".
Understand "Area 48 in Room 4" as a48r4.
the parent of the a48r4 is r4.
the X of the a48r4 is "548983.1511123565".
the Y of the a48r4 is "1578067.8660087993".

[create a49r4]
a49r4 is a area. "An area (49) in r4".
the printed name of the a49r4 is "Room 4".
Understand "Area 49 in Room 4" as a49r4.
the parent of the a49r4 is r4.
the X of the a49r4 is "548934.0".
the Y of the a49r4 is "1578125.0".

[create a50r4]
a50r4 is a area. "An area (50) in r4".
the printed name of the a50r4 is "Room 4".
Understand "Area 50 in Room 4" as a50r4.
the parent of the a50r4 is r4.
the X of the a50r4 is "548946.5833538161".
the Y of the a50r4 is "1578097.4302261274".

[create a71r4]
a71r4 is a area. "An area (71) in r4".
the printed name of the a71r4 is "Room 4".
Understand "Area 71 in Room 4" as a71r4.
the parent of the a71r4 is r4.
the X of the a71r4 is "548943.0".
the Y of the a71r4 is "1577600.0".

[create a77r4]
a77r4 is a area. "An area (77) in r4".
the printed name of the a77r4 is "Room 4".
Understand "Area 77 in Room 4" as a77r4.
the parent of the a77r4 is r4.
the X of the a77r4 is "547354.0219775602".
the Y of the a77r4 is "1577704.628948694".

[create a78r4]
a78r4 is a area. "An area (78) in r4".
the printed name of the a78r4 is "Room 4".
Understand "Area 78 in Room 4" as a78r4.
the parent of the a78r4 is r4.
the X of the a78r4 is "548962.5885022805".
the Y of the a78r4 is "1578061.5003010032".

[create d0]
d0 is a door. "A door between a8r3 and a0r4".
d0 is southwest of a8r3 and northeast of a0r4.

[create d2]
d2 is a door. "A door between a15r0 and a75r1".
d2 is southeast of a15r0 and northwest of a75r1.

[create d4]
d4 is a door. "A door between a12r2 and a13r3".
d4 is northeast of a12r2 and southwest of a13r3.

[create d6]
d6 is a door. "A door between a6r0 and a12r2".
d6 is southeast of a6r0 and northwest of a12r2.

east of a0r4 is west of a1r4.

southeast of a1r4 is northwest of a2r4.

northeast of a3r1 is southwest of a4r1.

east of a5r0 is west of a6r0.

east of a7r3 is west of a8r3.

east of a9r3 is west of a10r3.

west of a7r3 is east of a11r3.

south of a14r0 is north of a15r0.

southeast of a13r3 is northwest of a16r3.

southwest of a5r0 is northeast of a14r0.

south of a10r3 is north of a13r3.

northeast of a8r3 is southwest of a17r3.

east of a6r0 is west of a18r0.

southeast of a19r1 is northwest of a20r1.

south of a21r1 is north of a22r1.

southwest of a23r1 is northeast of a24r1.

northeast of a25r0 is southwest of a26r0.

east of a27r3 is west of a28r3.

north of a29r3 is south of a30r3.

southeast of a31r0 is northwest of a32r0.

west of a33r3 is east of a34r3.

northeast of a35r3 is southwest of a36r3.

northwest of a37r2 is southeast of a38r2.

southwest of a39r2 is northeast of a40r2.

north of a41r1 is south of a42r1.

east of a43r4 is west of a44r4.

northeast of a45r4 is southwest of a46r4.

southwest of a47r4 is northeast of a48r4.

southeast of a49r4 is northwest of a50r4.

northwest of a51r1 is southeast of a52r1.

northeast of a53r2 is southwest of a54r2.

southeast of a55r2 is northwest of a56r2.

northwest of a57r0 is southeast of a58r0.

southwest of a59r3 is northeast of a60r3.

southeast of a61r3 is northwest of a62r3.

southwest of a63r0 is northeast of a64r0.

south of a65r0 is north of a66r0.

west of a67r3 is east of a68r3.

east of a69r3 is west of a70r3.

southeast of a2r4 is northwest of a71r4.

northeast of a72r1 is southwest of a73r1.

southwest of a9r3 is northeast of a36r3.

southeast of a18r0 is northwest of a58r0.

northeast of a17r3 is southwest of a60r3.

northeast of a10r3 is southwest of a11r3.

northwest of a9r3 is southeast of a62r3.

northeast of a18r0 is southwest of a64r0.

north of a5r0 is south of a66r0.

north of a7r3 is south of a68r3.

north of a11r3 is south of a70r3.

southeast of a20r1 is northwest of a74r1.

southwest of a22r1 is northeast of a74r1.

southwest of a24r1 is northeast of a75r1.

southwest of a15r0 is northeast of a26r0.

southwest of a16r3 is northeast of a28r3.

southeast of a16r3 is northwest of a30r3.

west of a14r0 is east of a32r0.

east of a17r3 is west of a34r3.

northwest of a38r2 is southeast of a76r2.

southwest of a40r2 is northeast of a76r2.

south of a3r1 is north of a42r1.

east of a44r4 is west of a77r4.

northeast of a46r4 is southwest of a77r4.

west of a48r4 is east of a78r4.

southeast of a50r4 is northwest of a78r4.

southeast of a4r1 is northwest of a52r1.

northeast of a54r2 is southwest of a79r2.

southeast of a56r2 is northwest of a79r2.

southwest of a0r4 is northeast of a77r4.

northeast of a73r1 is southwest of a80r1.

northeast of a1r4 is southwest of a78r4.

south of a74r1 is north of a80r1.

south of a12r2 is north of a81r2.

north of a4r1 is south of a75r1.

west of a76r2 is east of a81r2.

east of a79r2 is west of a81r2.

west of a3r1 is east of a80r1.

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
the player is in a25r0.

the orientation of the player is 6.
