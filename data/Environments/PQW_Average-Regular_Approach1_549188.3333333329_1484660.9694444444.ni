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
the X of the a0r0 is "548537.0".
the Y of the a0r0 is "1484714.0".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "548506.1050372066".
the Y of the a1r0 is "1484687.7894485786".

[create a2r0]
a2r0 is a area. "An area (2) in r0".
the printed name of the a2r0 is "Room 0".
Understand "Area 2 in Room 0" as a2r0.
the parent of the a2r0 is r0.
the X of the a2r0 is "545120.0".
the Y of the a2r0 is "1484714.0".

[create a3r0]
a3r0 is a area. "An area (3) in r0".
the printed name of the a3r0 is "Room 0".
Understand "Area 3 in Room 0" as a3r0.
the parent of the a3r0 is r0.
the X of the a3r0 is "545150.9153780071".
the Y of the a3r0 is "1484688.1834621993".

[create a8r0]
a8r0 is a area. "An area (8) in r0".
the printed name of the a8r0 is "Room 0".
Understand "Area 8 in Room 0" as a8r0.
the parent of the a8r0 is r0.
the X of the a8r0 is "547811.6603924709".
the Y of the a8r0 is "1484708.965959151".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "547744.0".
the Y of the a9r0 is "1484679.6438392736".

[create a10r0]
a10r0 is a area. "An area (10) in r0".
the printed name of the a10r0 is "Room 0".
Understand "Area 10 in Room 0" as a10r0.
the parent of the a10r0 is r0.
the X of the a10r0 is "546693.0".
the Y of the a10r0 is "1483878.0".

[create a11r0]
a11r0 is a area. "An area (11) in r0".
the printed name of the a11r0 is "Room 0".
Understand "Area 11 in Room 0" as a11r0.
the parent of the a11r0 is r0.
the X of the a11r0 is "546604.6669468379".
the Y of the a11r0 is "1483917.5250000001".

[create a12r0]
a12r0 is a area. "An area (12) in r0".
the printed name of the a12r0 is "Room 0".
Understand "Area 12 in Room 0" as a12r0.
the parent of the a12r0 is r0.
the X of the a12r0 is "546542.0".
the Y of the a12r0 is "1484545.0".

[create a13r0]
a13r0 is a area. "An area (13) in r0".
the printed name of the a13r0 is "Room 0".
Understand "Area 13 in Room 0" as a13r0.
the parent of the a13r0 is r0.
the X of the a13r0 is "546608.9919958273".
the Y of the a13r0 is "1484480.5414679898".

[create a14r0]
a14r0 is a area. "An area (14) in r0".
the printed name of the a14r0 is "Room 0".
Understand "Area 14 in Room 0" as a14r0.
the parent of the a14r0 is r0.
the X of the a14r0 is "546336.0".
the Y of the a14r0 is "1484704.0".

[create a15r0]
a15r0 is a area. "An area (15) in r0".
the printed name of the a15r0 is "Room 0".
Understand "Area 15 in Room 0" as a15r0.
the parent of the a15r0 is r0.
the X of the a15r0 is "546302.2".
the Y of the a15r0 is "1484618.3716793556".

[create a16r0]
a16r0 is a area. "An area (16) in r0".
the printed name of the a16r0 is "Room 0".
Understand "Area 16 in Room 0" as a16r0.
the parent of the a16r0 is r0.
the X of the a16r0 is "545111.0".
the Y of the a16r0 is "1484624.0".

[create a17r0]
a17r0 is a area. "An area (17) in r0".
the printed name of the a17r0 is "Room 0".
Understand "Area 17 in Room 0" as a17r0.
the parent of the a17r0 is r0.
the X of the a17r0 is "545153.06".
the Y of the a17r0 is "1484642.519".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "548547.0".
the Y of the a22r0 is "1484624.0".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "548506.6686930091".
the Y of the a23r0 is "1484642.2965214455".

[create a40r0]
a40r0 is a area. "An area (40) in r0".
the printed name of the a40r0 is "Room 0".
Understand "Area 40 in Room 0" as a40r0.
the parent of the a40r0 is r0.
the X of the a40r0 is "546155.0".
the Y of the a40r0 is "1484038.0".

[create a41r0]
a41r0 is a area. "An area (41) in r0".
the printed name of the a41r0 is "Room 0".
Understand "Area 41 in Room 0" as a41r0.
the parent of the a41r0 is r0.
the X of the a41r0 is "546253.4772373155".
the Y of the a41r0 is "1484004.629844961".

[create a62r0]
a62r0 is a area. "An area (62) in r0".
the printed name of the a62r0 is "Room 0".
Understand "Area 62 in Room 0" as a62r0.
the parent of the a62r0 is r0.
the X of the a62r0 is "548106.8028395298".
the Y of the a62r0 is "1483382.5934021068".

[create a63r0]
a63r0 is a area. "An area (63) in r0".
the printed name of the a63r0 is "Room 0".
Understand "Area 63 in Room 0" as a63r0.
the parent of the a63r0 is r0.
the X of the a63r0 is "546406.0788892417".
the Y of the a63r0 is "1483953.1686115644".

[create a64r0]
a64r0 is a area. "An area (64) in r0".
the printed name of the a64r0 is "Room 0".
Understand "Area 64 in Room 0" as a64r0.
the parent of the a64r0 is r0.
the X of the a64r0 is "546438.3208999769".
the Y of the a64r0 is "1483986.6212089316".

[create a65r0]
a65r0 is a area. "An area (65) in r0".
the printed name of the a65r0 is "Room 0".
Understand "Area 65 in Room 0" as a65r0.
the parent of the a65r0 is r0.
the X of the a65r0 is "546624.1760514181".
the Y of the a65r0 is "1484455.0491483654".

[create a66r0]
a66r0 is a area. "An area (66) in r0".
the printed name of the a66r0 is "Room 0".
Understand "Area 66 in Room 0" as a66r0.
the parent of the a66r0 is r0.
the X of the a66r0 is "546291.892037503".
the Y of the a66r0 is "1484591.1972183192".

[create a67r0]
a67r0 is a area. "An area (67) in r0".
the printed name of the a67r0 is "Room 0".
Understand "Area 67 in Room 0" as a67r0.
the parent of the a67r0 is r0.
the X of the a67r0 is "545168.0926719012".
the Y of the a67r0 is "1484659.8065726864".

[create a75r0]
a75r0 is a area. "An area (75) in r0".
the printed name of the a75r0 is "Room 0".
Understand "Area 75 in Room 0" as a75r0.
the parent of the a75r0 is r0.
the X of the a75r0 is "547741.5112273855".
the Y of the a75r0 is "1484672.2799400566".

[create a77r0]
a77r0 is a area. "An area (77) in r0".
the printed name of the a77r0 is "Room 0".
Understand "Area 77 in Room 0" as a77r0.
the parent of the a77r0 is r0.
the X of the a77r0 is "548490.1653756483".
the Y of the a77r0 is "1484661.2658517454".

[create a83r0]
a83r0 is a area. "An area (83) in r0".
the printed name of the a83r0 is "Room 0".
Understand "Area 83 in Room 0" as a83r0.
the parent of the a83r0 is r0.
the X of the a83r0 is "545531.1030054238".
the Y of the a83r0 is "1484530.557038129".

[create a84r0]
a84r0 is a area. "An area (84) in r0".
the printed name of the a84r0 is "Room 0".
Understand "Area 84 in Room 0" as a84r0.
the parent of the a84r0 is r0.
the X of the a84r0 is "548111.0866855372".
the Y of the a84r0 is "1484539.5872257592".

[create a85r0]
a85r0 is a area. "An area (85) in r0".
the printed name of the a85r0 is "Room 0".
Understand "Area 85 in Room 0" as a85r0.
the parent of the a85r0 is r0.
the X of the a85r0 is "546303.8313347906".
the Y of the a85r0 is "1483548.7790697673".

[create a86r0]
a86r0 is a area. "An area (86) in r0".
the printed name of the a86r0 is "Room 0".
Understand "Area 86 in Room 0" as a86r0.
the parent of the a86r0 is r0.
the X of the a86r0 is "548215.0376784218".
the Y of the a86r0 is "1483600.473885267".

[create a87r0]
a87r0 is a area. "An area (87) in r0".
the printed name of the a87r0 is "Room 0".
Understand "Area 87 in Room 0" as a87r0.
the parent of the a87r0 is r0.
the X of the a87r0 is "546545.1819171563".
the Y of the a87r0 is "1484378.8405485917".

[create a88r0]
a88r0 is a area. "An area (88) in r0".
the printed name of the a88r0 is "Room 0".
Understand "Area 88 in Room 0" as a88r0.
the parent of the a88r0 is r0.
the X of the a88r0 is "545531.7860830325".
the Y of the a88r0 is "1483538.5".

[create a32r1]
a32r1 is a area. "An area (32) in r1".
the printed name of the a32r1 is "Room 1".
Understand "Area 32 in Room 1" as a32r1.
the parent of the a32r1 is r1.
the X of the a32r1 is "546643.0".
the Y of the a32r1 is "1482551.0".

[create a33r1]
a33r1 is a area. "An area (33) in r1".
the printed name of the a33r1 is "Room 1".
Understand "Area 33 in Room 1" as a33r1.
the parent of the a33r1 is r1.
the X of the a33r1 is "546679.0".
the Y of the a33r1 is "1482632.3823529412".

[create a34r1]
a34r1 is a area. "An area (34) in r1".
the printed name of the a34r1 is "Room 1".
Understand "Area 34 in Room 1" as a34r1.
the parent of the a34r1 is r1.
the X of the a34r1 is "547887.0".
the Y of the a34r1 is "1482701.0".

[create a35r1]
a35r1 is a area. "An area (35) in r1".
the printed name of the a35r1 is "Room 1".
Understand "Area 35 in Room 1" as a35r1.
the parent of the a35r1 is r1.
the X of the a35r1 is "547831.991511449".
the Y of the a35r1 is "1482777.2877525997".

[create a36r1]
a36r1 is a area. "An area (36) in r1".
the printed name of the a36r1 is "Room 1".
Understand "Area 36 in Room 1" as a36r1.
the parent of the a36r1 is r1.
the X of the a36r1 is "546155.0".
the Y of the a36r1 is "1483095.0".

[create a37r1]
a37r1 is a area. "An area (37) in r1".
the printed name of the a37r1 is "Room 1".
Understand "Area 37 in Room 1" as a37r1.
the parent of the a37r1 is r1.
the X of the a37r1 is "546250.4098360653".
the Y of the a37r1 is "1483131.5".

[create a38r1]
a38r1 is a area. "An area (38) in r1".
the printed name of the a38r1 is "Room 1".
Understand "Area 38 in Room 1" as a38r1.
the parent of the a38r1 is r1.
the X of the a38r1 is "548268.0".
the Y of the a38r1 is "1483334.6347305388".

[create a39r1]
a39r1 is a area. "An area (39) in r1".
the printed name of the a39r1 is "Room 1".
Understand "Area 39 in Room 1" as a39r1.
the parent of the a39r1 is r1.
the X of the a39r1 is "548193.8582936489".
the Y of the a39r1 is "1483332.8173652694".

[create a42r1]
a42r1 is a area. "An area (42) in r1".
the printed name of the a42r1 is "Room 1".
Understand "Area 42 in Room 1" as a42r1.
the parent of the a42r1 is r1.
the X of the a42r1 is "548207.0".
the Y of the a42r1 is "1481933.0".

[create a43r1]
a43r1 is a area. "An area (43) in r1".
the printed name of the a43r1 is "Room 1".
Understand "Area 43 in Room 1" as a43r1.
the parent of the a43r1 is r1.
the X of the a43r1 is "548211.2013157895".
the Y of the a43r1 is "1482006.8065789475".

[create a46r1]
a46r1 is a area. "An area (46) in r1".
the printed name of the a46r1 is "Room 1".
Understand "Area 46 in Room 1" as a46r1.
the parent of the a46r1 is r1.
the X of the a46r1 is "548462.0".
the Y of the a46r1 is "1482121.0".

[create a47r1]
a47r1 is a area. "An area (47) in r1".
the printed name of the a47r1 is "Room 1".
Understand "Area 47 in Room 1" as a47r1.
the parent of the a47r1 is r1.
the X of the a47r1 is "548407.0661764707".
the Y of the a47r1 is "1482087.0".

[create a48r1]
a48r1 is a area. "An area (48) in r1".
the printed name of the a48r1 is "Room 1".
Understand "Area 48 in Room 1" as a48r1.
the parent of the a48r1 is r1.
the X of the a48r1 is "548467.0".
the Y of the a48r1 is "1482572.0".

[create a49r1]
a49r1 is a area. "An area (49) in r1".
the printed name of the a49r1 is "Room 1".
Understand "Area 49 in Room 1" as a49r1.
the parent of the a49r1 is r1.
the X of the a49r1 is "548497.7567299004".
the Y of the a49r1 is "1482620.7536488604".

[create a52r1]
a52r1 is a area. "An area (52) in r1".
the printed name of the a52r1 is "Room 1".
Understand "Area 52 in Room 1" as a52r1.
the parent of the a52r1 is r1.
the X of the a52r1 is "548749.0".
the Y of the a52r1 is "1482600.0".

[create a53r1]
a53r1 is a area. "An area (53) in r1".
the printed name of the a53r1 is "Room 1".
Understand "Area 53 in Room 1" as a53r1.
the parent of the a53r1 is r1.
the X of the a53r1 is "548708.2968536066".
the Y of the a53r1 is "1482651.4209743899".

[create a58r1]
a58r1 is a area. "An area (58) in r1".
the printed name of the a58r1 is "Room 1".
Understand "Area 58 in Room 1" as a58r1.
the parent of the a58r1 is r1.
the X of the a58r1 is "548600.9282688181".
the Y of the a58r1 is "1482764.0202992572".

[create a59r1]
a59r1 is a area. "An area (59) in r1".
the printed name of the a59r1 is "Room 1".
Understand "Area 59 in Room 1" as a59r1.
the parent of the a59r1 is r1.
the X of the a59r1 is "548614.7496886395".
the Y of the a59r1 is "1482762.3173736061".

[create a60r1]
a60r1 is a area. "An area (60) in r1".
the printed name of the a60r1 is "Room 1".
Understand "Area 60 in Room 1" as a60r1.
the parent of the a60r1 is r1.
the X of the a60r1 is "547800.025339793".
the Y of the a60r1 is "1482817.9152977373".

[create a61r1]
a61r1 is a area. "An area (61) in r1".
the printed name of the a61r1 is "Room 1".
Understand "Area 61 in Room 1" as a61r1.
the parent of the a61r1 is r1.
the X of the a61r1 is "546347.621359989".
the Y of the a61r1 is "1483168.7364393054".

[create a68r1]
a68r1 is a area. "An area (68) in r1".
the printed name of the a68r1 is "Room 1".
Understand "Area 68 in Room 1" as a68r1.
the parent of the a68r1 is r1.
the X of the a68r1 is "548219.7952866211".
the Y of the a68r1 is "1482090.5977945551".

[create a71r1]
a71r1 is a area. "An area (71) in r1".
the printed name of the a71r1 is "Room 1".
Understand "Area 71 in Room 1" as a71r1.
the parent of the a71r1 is r1.
the X of the a71r1 is "546707.2997365496".
the Y of the a71r1 is "1482695.2244149854".

[create a79r1]
a79r1 is a area. "An area (79) in r1".
the printed name of the a79r1 is "Room 1".
Understand "Area 79 in Room 1" as a79r1.
the parent of the a79r1 is r1.
the X of the a79r1 is "547385.8480141505".
the Y of the a79r1 is "1482583.1248914613".

[create a24r2]
a24r2 is a area. "An area (24) in r2".
the printed name of the a24r2 is "Room 2".
Understand "Area 24 in Room 2" as a24r2.
the parent of the a24r2 is r2.
the X of the a24r2 is "549355.0".
the Y of the a24r2 is "1484704.0".

[create a25r2]
a25r2 is a area. "An area (25) in r2".
the printed name of the a25r2 is "Room 2".
Understand "Area 25 in Room 2" as a25r2.
the parent of the a25r2 is r2.
the X of the a25r2 is "549321.6666666665".
the Y of the a25r2 is "1484653.5364495006".

[create a56r2]
a56r2 is a area. "An area (56) in r2".
the printed name of the a56r2 is "Room 2".
Understand "Area 56 in Room 2" as a56r2.
the parent of the a56r2 is r2.
the X of the a56r2 is "549155.0".
the Y of the a56r2 is "1484704.0".

[create a57r2]
a57r2 is a area. "An area (57) in r2".
the printed name of the a57r2 is "Room 2".
Understand "Area 57 in Room 2" as a57r2.
the parent of the a57r2 is r2.
the X of the a57r2 is "549188.3333333329".
the Y of the a57r2 is "1484660.9694444444".

[create a78r2]
a78r2 is a area. "An area (78) in r2".
the printed name of the a78r2 is "Room 2".
Understand "Area 78 in Room 2" as a78r2.
the parent of the a78r2 is r2.
the X of the a78r2 is "549255.5167685037".
the Y of the a78r2 is "1484573.513395759".

[create a81r2]
a81r2 is a area. "An area (81) in r2".
the printed name of the a81r2 is "Room 2".
Understand "Area 81 in Room 2" as a81r2.
the parent of the a81r2 is r2.
the X of the a81r2 is "549260.8708448358".
the Y of the a81r2 is "1484405.3162913683".

[create a50r3]
a50r3 is a area. "An area (50) in r3".
the printed name of the a50r3 is "Room 3".
Understand "Area 50 in Room 3" as a50r3.
the parent of the a50r3 is r3.
the X of the a50r3 is "549729.0".
the Y of the a50r3 is "1482572.0".

[create a51r3]
a51r3 is a area. "An area (51) in r3".
the printed name of the a51r3 is "Room 3".
Understand "Area 51 in Room 3" as a51r3.
the parent of the a51r3 is r3.
the X of the a51r3 is "549759.7043713275".
the Y of the a51r3 is "1482622.706711225".

[create a54r3]
a54r3 is a area. "An area (54) in r3".
the printed name of the a54r3 is "Room 3".
Understand "Area 54 in Room 3" as a54r3.
the parent of the a54r3 is r3.
the X of the a54r3 is "550012.0".
the Y of the a54r3 is "1482600.0".

[create a55r3]
a55r3 is a area. "An area (55) in r3".
the printed name of the a55r3 is "Room 3".
Understand "Area 55 in Room 3" as a55r3.
the parent of the a55r3 is r3.
the X of the a55r3 is "549971.4539705466".
the Y of the a55r3 is "1482648.7643334027".

[create a72r3]
a72r3 is a area. "An area (72) in r3".
the printed name of the a72r3 is "Room 3".
Understand "Area 72 in Room 3" as a72r3.
the parent of the a72r3 is r3.
the X of the a72r3 is "549855.2390740267".
the Y of the a72r3 is "1482762.431282769".

[create a73r3]
a73r3 is a area. "An area (73) in r3".
the printed name of the a73r3 is "Room 3".
Understand "Area 73 in Room 3" as a73r3.
the parent of the a73r3 is r3.
the X of the a73r3 is "549877.2896516463".
the Y of the a73r3 is "1482760.8833195835".

[create a4r4]
a4r4 is a area. "An area (4) in r4".
the printed name of the a4r4 is "Room 4".
Understand "Area 4 in Room 4" as a4r4.
the parent of the a4r4 is r4.
the X of the a4r4 is "550006.0".
the Y of the a4r4 is "1484714.0".

[create a5r4]
a5r4 is a area. "An area (5) in r4".
the printed name of the a5r4 is "Room 4".
Understand "Area 5 in Room 4" as a5r4.
the parent of the a5r4 is r4.
the X of the a5r4 is "550036.9153780069".
the Y of the a5r4 is "1484688.1834621993".

[create a6r4]
a6r4 is a area. "An area (6) in r4".
the printed name of the a6r4 is "Room 4".
Understand "Area 6 in Room 4" as a6r4.
the parent of the a6r4 is r4.
the X of the a6r4 is "550765.0".
the Y of the a6r4 is "1484714.0".

[create a7r4]
a7r4 is a area. "An area (7) in r4".
the printed name of the a7r4 is "Room 4".
Understand "Area 7 in Room 4" as a7r4.
the parent of the a7r4 is r4.
the X of the a7r4 is "550733.9123931623".
the Y of the a7r4 is "1484688.1662393163".

[create a18r4]
a18r4 is a area. "An area (18) in r4".
the printed name of the a18r4 is "Room 4".
Understand "Area 18 in Room 4" as a18r4.
the parent of the a18r4 is r4.
the X of the a18r4 is "549997.0".
the Y of the a18r4 is "1484624.0".

[create a19r4]
a19r4 is a area. "An area (19) in r4".
the printed name of the a19r4 is "Room 4".
Understand "Area 19 in Room 4" as a19r4.
the parent of the a19r4 is r4.
the X of the a19r4 is "550037.7622699386".
the Y of the a19r4 is "1484642.6487730062".

[create a20r4]
a20r4 is a area. "An area (20) in r4".
the printed name of the a20r4 is "Room 4".
Understand "Area 20 in Room 4" as a20r4.
the parent of the a20r4 is r4.
the X of the a20r4 is "550774.0".
the Y of the a20r4 is "1484624.0".

[create a21r4]
a21r4 is a area. "An area (21) in r4".
the printed name of the a21r4 is "Room 4".
Understand "Area 21 in Room 4" as a21r4.
the parent of the a21r4 is r4.
the X of the a21r4 is "550733.6625777471".
the Y of the a21r4 is "1484642.6912577746".

[create a74r4]
a74r4 is a area. "An area (74) in r4".
the printed name of the a74r4 is "Room 4".
Understand "Area 74 in Room 4" as a74r4.
the parent of the a74r4 is r4.
the X of the a74r4 is "550717.5509290382".
the Y of the a74r4 is "1484661.0062088708".

[create a76r4]
a76r4 is a area. "An area (76) in r4".
the printed name of the a76r4 is "Room 4".
Understand "Area 76 in Room 4" as a76r4.
the parent of the a76r4 is r4.
the X of the a76r4 is "550053.4921500897".
the Y of the a76r4 is "1484660.7986347189".

[create a80r4]
a80r4 is a area. "An area (80) in r4".
the printed name of the a80r4 is "Room 4".
Understand "Area 80 in Room 4" as a80r4.
the parent of the a80r4 is r4.
the X of the a80r4 is "550387.9486065779".
the Y of the a80r4 is "1484578.1064398696".

[create a26r5]
a26r5 is a area. "An area (26) in r5".
the printed name of the a26r5 is "Room 5".
Understand "Area 26 in Room 5" as a26r5.
the parent of the a26r5 is r5.
the X of the a26r5 is "545089.0".
the Y of the a26r5 is "1482680.0".

[create a27r5]
a27r5 is a area. "An area (27) in r5".
the printed name of the a27r5 is "Room 5".
Understand "Area 27 in Room 5" as a27r5.
the parent of the a27r5 is r5.
the X of the a27r5 is "545120.0059757694".
the Y of the a27r5 is "1482660.657907662".

[create a28r5]
a28r5 is a area. "An area (28) in r5".
the printed name of the a28r5 is "Room 5".
Understand "Area 28 in Room 5" as a28r5.
the parent of the a28r5 is r5.
the X of the a28r5 is "545940.0".
the Y of the a28r5 is "1482591.0".

[create a29r5]
a29r5 is a area. "An area (29) in r5".
the printed name of the a29r5 is "Room 5".
Understand "Area 29 in Room 5" as a29r5.
the parent of the a29r5 is r5.
the X of the a29r5 is "545908.7862838916".
the Y of the a29r5 is "1482616.6339712918".

[create a30r5]
a30r5 is a area. "An area (30) in r5".
the printed name of the a30r5 is "Room 5".
Understand "Area 30 in Room 5" as a30r5.
the parent of the a30r5 is r5.
the X of the a30r5 is "545949.0".
the Y of the a30r5 is "1482680.0".

[create a31r5]
a31r5 is a area. "An area (31) in r5".
the printed name of the a31r5 is "Room 5".
Understand "Area 31 in Room 5" as a31r5.
the parent of the a31r5 is r5.
the X of the a31r5 is "545906.1485252036".
the Y of the a31r5 is "1482661.855767114".

[create a44r5]
a44r5 is a area. "An area (44) in r5".
the printed name of the a44r5 is "Room 5".
Understand "Area 44 in Room 5" as a44r5.
the parent of the a44r5 is r5.
the X of the a44r5 is "545098.0".
the Y of the a44r5 is "1482591.0".

[create a45r5]
a45r5 is a area. "An area (45) in r5".
the printed name of the a45r5 is "Room 5".
Understand "Area 45 in Room 5" as a45r5.
the parent of the a45r5 is r5.
the X of the a45r5 is "545129.2137161086".
the Y of the a45r5 is "1482616.6339712918".

[create a69r5]
a69r5 is a area. "An area (69) in r5".
the printed name of the a69r5 is "Room 5".
Understand "Area 69 in Room 5" as a69r5.
the parent of the a69r5 is r5.
the X of the a69r5 is "545143.389199255".
the Y of the a69r5 is "1482640.2597765364".

[create a70r5]
a70r5 is a area. "An area (70) in r5".
the printed name of the a70r5 is "Room 5".
Understand "Area 70 in Room 5" as a70r5.
the parent of the a70r5 is r5.
the X of the a70r5 is "545891.7817693062".
the Y of the a70r5 is "1482644.9748289345".

[create a82r5]
a82r5 is a area. "An area (82) in r5".
the printed name of the a82r5 is "Room 5".
Understand "Area 82 in Room 5" as a82r5.
the parent of the a82r5 is r5.
the X of the a82r5 is "545525.0".
the Y of the a82r5 is "1482768.577830189".

[create d0]
d0 is a door. "A door between a62r0 and a39r1".
d0 is southeast of a62r0 and northwest of a39r1.

[create d1]
d1 is a door. "A door between a86r0 and a58r1".
d1 is southeast of a86r0 and northwest of a58r1.

[create d2]
d2 is a door. "A door between a73r3 and a80r4".
d2 is north of a73r3 and south of a80r4.

[create d4]
d4 is a door. "A door between a59r1 and a81r2".
d4 is north of a59r1 and south of a81r2.

[create d6]
d6 is a door. "A door between a81r2 and a72r3".
d6 is southeast of a81r2 and northwest of a72r3.

[create d8]
d8 is a door. "A door between a62r0 and a60r1".
d8 is southwest of a62r0 and northeast of a60r1.

[create d9]
d9 is a door. "A door between a85r0 and a61r1".
d9 is south of a85r0 and north of a61r1.

[create d10]
d10 is a door. "A door between a88r0 and a82r5".
d10 is south of a88r0 and north of a82r5.

southwest of a0r0 is northeast of a1r0.

southeast of a2r0 is northwest of a3r0.

southeast of a4r4 is northwest of a5r4.

southwest of a6r4 is northeast of a7r4.

southwest of a8r0 is northeast of a9r0.

northwest of a10r0 is southeast of a11r0.

southeast of a12r0 is northwest of a13r0.

south of a14r0 is north of a15r0.

northeast of a16r0 is southwest of a17r0.

northeast of a18r4 is southwest of a19r4.

northwest of a20r4 is southeast of a21r4.

northwest of a22r0 is southeast of a23r0.

southwest of a24r2 is northeast of a25r2.

southeast of a26r5 is northwest of a27r5.

northwest of a28r5 is southeast of a29r5.

southwest of a30r5 is northeast of a31r5.

northeast of a32r1 is southwest of a33r1.

northwest of a34r1 is southeast of a35r1.

east of a36r1 is west of a37r1.

west of a38r1 is east of a39r1.

east of a40r0 is west of a41r0.

north of a42r1 is south of a43r1.

northeast of a44r5 is southwest of a45r5.

southwest of a46r1 is northeast of a47r1.

northeast of a48r1 is southwest of a49r1.

northeast of a50r3 is southwest of a51r3.

northwest of a52r1 is southeast of a53r1.

northwest of a54r3 is southeast of a55r3.

southeast of a56r2 is northwest of a57r2.

east of a58r1 is west of a59r1.

northwest of a35r1 is southeast of a60r1.

east of a37r1 is west of a61r1.

east of a41r0 is west of a63r0.

northwest of a11r0 is southeast of a64r0.

southeast of a13r0 is northwest of a65r0.

south of a15r0 is north of a66r0.

northeast of a17r0 is southwest of a67r0.

north of a43r1 is south of a68r1.

northeast of a45r5 is southwest of a69r5.

southeast of a27r5 is northwest of a69r5.

northwest of a29r5 is southeast of a70r5.

southwest of a31r5 is northeast of a70r5.

northeast of a33r1 is southwest of a71r1.

east of a72r3 is west of a73r3.

southwest of a7r4 is northeast of a74r4.

south of a9r0 is north of a75r0.

northeast of a19r4 is southwest of a76r4.

northwest of a21r4 is southeast of a74r4.

northwest of a23r0 is southeast of a77r0.

southwest of a25r2 is northeast of a78r2.

southwest of a1r0 is northeast of a77r0.

southeast of a3r0 is northwest of a67r0.

southeast of a5r4 is northwest of a76r4.

southwest of a60r1 is northeast of a79r1.

northeast of a49r1 is southwest of a58r1.

northeast of a51r3 is southwest of a72r3.

southeast of a61r1 is northwest of a71r1.

northwest of a53r1 is southeast of a59r1.

northwest of a55r3 is southeast of a73r3.

west of a47r1 is east of a68r1.

northwest of a68r1 is southeast of a79r1.

west of a70r5 is east of a82r5.

east of a69r5 is west of a82r5.

east of a71r1 is west of a79r1.

west of a66r0 is east of a83r0.

east of a67r0 is west of a83r0.

southeast of a57r2 is northwest of a78r2.

east of a75r0 is west of a84r0.

west of a77r0 is east of a84r0.

east of a76r4 is west of a80r4.

west of a74r4 is east of a80r4.

south of a63r0 is north of a85r0.

south of a84r0 is north of a86r0.

northeast of a63r0 is southwest of a64r0.

north of a64r0 is south of a87r0.

southwest of a65r0 is northeast of a87r0.

southeast of a66r0 is northwest of a87r0.

east of a65r0 is west of a75r0.

south of a78r2 is north of a81r2.

west of a85r0 is east of a88r0.

northeast of a62r0 is southwest of a86r0.

south of a83r0 is north of a88r0.

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
the player is in a57r2.

the orientation of the player is 6.
