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

[create a7r0]
a7r0 is a area. "An area (7) in r0".
the printed name of the a7r0 is "Room 0".
Understand "Area 7 in Room 0" as a7r0.
the parent of the a7r0 is r0.
the X of the a7r0 is "547063.9898805368".
the Y of the a7r0 is "2522075.007280613".

[create a11r0]
a11r0 is a area. "An area (11) in r0".
the printed name of the a11r0 is "Room 0".
Understand "Area 11 in Room 0" as a11r0.
the parent of the a11r0 is r0.
the X of the a11r0 is "546892.795011837".
the Y of the a11r0 is "2522049.780342808".

[create a26r0]
a26r0 is a area. "An area (26) in r0".
the printed name of the a26r0 is "Room 0".
Understand "Area 26 in Room 0" as a26r0.
the parent of the a26r0 is r0.
the X of the a26r0 is "545148.7".
the Y of the a26r0 is "2522217.2770407326".

[create a27r0]
a27r0 is a area. "An area (27) in r0".
the printed name of the a27r0 is "Room 0".
Understand "Area 27 in Room 0" as a27r0.
the parent of the a27r0 is r0.
the X of the a27r0 is "545386.2064445314".
the Y of the a27r0 is "2522095.6709869187".

[create a28r0]
a28r0 is a area. "An area (28) in r0".
the printed name of the a28r0 is "Room 0".
Understand "Area 28 in Room 0" as a28r0.
the parent of the a28r0 is r0.
the X of the a28r0 is "546675.7499999998".
the Y of the a28r0 is "2522176.847084048".

[create a29r0]
a29r0 is a area. "An area (29) in r0".
the printed name of the a29r0 is "Room 0".
Understand "Area 29 in Room 0" as a29r0.
the parent of the a29r0 is r0.
the X of the a29r0 is "546730.3964885047".
the Y of the a29r0 is "2522085.630012937".

[create a30r0]
a30r0 is a area. "An area (30) in r0".
the printed name of the a30r0 is "Room 0".
Understand "Area 30 in Room 0" as a30r0.
the parent of the a30r0 is r0.
the X of the a30r0 is "547146.6666666673".
the Y of the a30r0 is "2522188.4059129027".

[create a33r0]
a33r0 is a area. "An area (33) in r0".
the printed name of the a33r0 is "Room 0".
Understand "Area 33 in Room 0" as a33r0.
the parent of the a33r0 is r0.
the X of the a33r0 is "546027.685593508".
the Y of the a33r0 is "2521230.6292565824".

[create a40r0]
a40r0 is a area. "An area (40) in r0".
the printed name of the a40r0 is "Room 0".
Understand "Area 40 in Room 0" as a40r0.
the parent of the a40r0 is r0.
the X of the a40r0 is "545461.0".
the Y of the a40r0 is "2522240.0".

[create a41r0]
a41r0 is a area. "An area (41) in r0".
the printed name of the a41r0 is "Room 0".
Understand "Area 41 in Room 0" as a41r0.
the parent of the a41r0 is r0.
the X of the a41r0 is "545426.3".
the Y of the a41r0 is "2522168.436524823".

[create a50r0]
a50r0 is a area. "An area (50) in r0".
the printed name of the a50r0 is "Room 0".
Understand "Area 50 in Room 0" as a50r0.
the parent of the a50r0 is r0.
the X of the a50r0 is "545114.0".
the Y of the a50r0 is "2522240.0".

[create a51r0]
a51r0 is a area. "An area (51) in r0".
the printed name of the a51r0 is "Room 0".
Understand "Area 51 in Room 0" as a51r0.
the parent of the a51r0 is r0.
the X of the a51r0 is "546653.0".
the Y of the a51r0 is "2522240.0".

[create a52r0]
a52r0 is a area. "An area (52) in r0".
the printed name of the a52r0 is "Room 0".
Understand "Area 52 in Room 0" as a52r0.
the parent of the a52r0 is r0.
the X of the a52r0 is "547182.0".
the Y of the a52r0 is "2522240.0".

[create a2r1]
a2r1 is a area. "An area (2) in r1".
the printed name of the a2r1 is "Room 1".
Understand "Area 2 in Room 1" as a2r1.
the parent of the a2r1 is r1.
the X of the a2r1 is "547813.1926899109".
the Y of the a2r1 is "2520157.8341229064".

[create a8r1]
a8r1 is a area. "An area (8) in r1".
the printed name of the a8r1 is "Room 1".
Understand "Area 8 in Room 1" as a8r1.
the parent of the a8r1 is r1.
the X of the a8r1 is "547772.6246433791".
the Y of the a8r1 is "2520159.8817214696".

[create a18r1]
a18r1 is a area. "An area (18) in r1".
the printed name of the a18r1 is "Room 1".
Understand "Area 18 in Room 1" as a18r1.
the parent of the a18r1 is r1.
the X of the a18r1 is "547693.75".
the Y of the a18r1 is "2520051.7428626004".

[create a19r1]
a19r1 is a area. "An area (19) in r1".
the printed name of the a19r1 is "Room 1".
Understand "Area 19 in Room 1" as a19r1.
the parent of the a19r1 is r1.
the X of the a19r1 is "547890.25".
the Y of the a19r1 is "2520051.484325186".

[create a46r1]
a46r1 is a area. "An area (46) in r1".
the printed name of the a46r1 is "Room 1".
Understand "Area 46 in Room 1" as a46r1.
the parent of the a46r1 is r1.
the X of the a46r1 is "547661.0".
the Y of the a46r1 is "2520000.0".

[create a47r1]
a47r1 is a area. "An area (47) in r1".
the printed name of the a47r1 is "Room 1".
Understand "Area 47 in Room 1" as a47r1.
the parent of the a47r1 is r1.
the X of the a47r1 is "547923.0".
the Y of the a47r1 is "2520000.0".

[create a3r2]
a3r2 is a area. "An area (3) in r2".
the printed name of the a3r2 is "Room 2".
Understand "Area 3 in Room 2" as a3r2.
the parent of the a3r2 is r2.
the X of the a3r2 is "548579.0511527918".
the Y of the a3r2 is "2522061.7482603416".

[create a23r2]
a23r2 is a area. "An area (23) in r2".
the printed name of the a23r2 is "Room 2".
Understand "Area 23 in Room 2" as a23r2.
the parent of the a23r2 is r2.
the X of the a23r2 is "548654.57021411".
the Y of the a23r2 is "2522050.429048625".

[create a24r2]
a24r2 is a area. "An area (24) in r2".
the printed name of the a24r2 is "Room 2".
Understand "Area 24 in Room 2" as a24r2.
the parent of the a24r2 is r2.
the X of the a24r2 is "548803.8000000003".
the Y of the a24r2 is "2522202.067061144".

[create a31r2]
a31r2 is a area. "An area (31) in r2".
the printed name of the a31r2 is "Room 2".
Understand "Area 31 in Room 2" as a31r2.
the parent of the a31r2 is r2.
the X of the a31r2 is "548490.2000000004".
the Y of the a31r2 is "2522185.378981202".

[create a53r2]
a53r2 is a area. "An area (53) in r2".
the printed name of the a53r2 is "Room 2".
Understand "Area 53 in Room 2" as a53r2.
the parent of the a53r2 is r2.
the X of the a53r2 is "548451.0".
the Y of the a53r2 is "2522240.0".

[create a58r2]
a58r2 is a area. "An area (58) in r2".
the printed name of the a58r2 is "Room 2".
Understand "Area 58 in Room 2" as a58r2.
the parent of the a58r2 is r2.
the X of the a58r2 is "548843.0".
the Y of the a58r2 is "2522240.0".

[create a0r3]
a0r3 is a area. "An area (0) in r3".
the printed name of the a0r3 is "Room 3".
Understand "Area 0 in Room 3" as a0r3.
the parent of the a0r3 is r3.
the X of the a0r3 is "550359.2".
the Y of the a0r3 is "2520054.950677507".

[create a1r3]
a1r3 is a area. "An area (1) in r3".
the printed name of the a1r3 is "Room 3".
Understand "Area 1 in Room 3" as a1r3.
the parent of the a1r3 is r3.
the X of the a1r3 is "550430.4331419051".
the Y of the a1r3 is "2520152.0435060626".

[create a4r3]
a4r3 is a area. "An area (4) in r3".
the printed name of the a4r3 is "Room 3".
Understand "Area 4 in Room 3" as a4r3.
the parent of the a4r3 is r3.
the X of the a4r3 is "550139.3099542343".
the Y of the a4r3 is "2520818.866232838".

[create a12r3]
a12r3 is a area. "An area (12) in r3".
the printed name of the a12r3 is "Room 3".
Understand "Area 12 in Room 3" as a12r3.
the parent of the a12r3 is r3.
the X of the a12r3 is "548765.5351623071".
the Y of the a12r3 is "2520205.0967777153".

[create a13r3]
a13r3 is a area. "An area (13) in r3".
the printed name of the a13r3 is "Room 3".
Understand "Area 13 in Room 3" as a13r3.
the parent of the a13r3 is r3.
the X of the a13r3 is "548803.6719556495".
the Y of the a13r3 is "2520407.845669572".

[create a16r3]
a16r3 is a area. "An area (16) in r3".
the printed name of the a16r3 is "Room 3".
Understand "Area 16 in Room 3" as a16r3.
the parent of the a16r3 is r3.
the X of the a16r3 is "548580.5".
the Y of the a16r3 is "2520039.2083333335".

[create a17r3]
a17r3 is a area. "An area (17) in r3".
the printed name of the a17r3 is "Room 3".
Understand "Area 17 in Room 3" as a17r3.
the parent of the a17r3 is r3.
the X of the a17r3 is "550624.8".
the Y of the a17r3 is "2520028.622119816".

[create a20r3]
a20r3 is a area. "An area (20) in r3".
the printed name of the a20r3 is "Room 3".
Understand "Area 20 in Room 3" as a20r3.
the parent of the a20r3 is r3.
the X of the a20r3 is "548876.0".
the Y of the a20r3 is "2520053.48670393".

[create a32r3]
a32r3 is a area. "An area (32) in r3".
the printed name of the a32r3 is "Room 3".
Understand "Area 32 in Room 3" as a32r3.
the parent of the a32r3 is r3.
the X of the a32r3 is "549086.4078084965".
the Y of the a32r3 is "2520812.9572670264".

[create a48r3]
a48r3 is a area. "An area (48) in r3".
the printed name of the a48r3 is "Room 3".
Understand "Area 48 in Room 3" as a48r3.
the parent of the a48r3 is r3.
the X of the a48r3 is "548909.0".
the Y of the a48r3 is "2520000.0".

[create a49r3]
a49r3 is a area. "An area (49) in r3".
the printed name of the a49r3 is "Room 3".
Understand "Area 49 in Room 3" as a49r3.
the parent of the a49r3 is r3.
the X of the a49r3 is "550326.0".
the Y of the a49r3 is "2520000.0".

[create a56r3]
a56r3 is a area. "An area (56) in r3".
the printed name of the a56r3 is "Room 3".
Understand "Area 56 in Room 3" as a56r3.
the parent of the a56r3 is r3.
the X of the a56r3 is "548599.0".
the Y of the a56r3 is "2520000.0".

[create a57r3]
a57r3 is a area. "An area (57) in r3".
the printed name of the a57r3 is "Room 3".
Understand "Area 57 in Room 3" as a57r3.
the parent of the a57r3 is r3.
the X of the a57r3 is "550658.0".
the Y of the a57r3 is "2520000.0".

[create a21r4]
a21r4 is a area. "An area (21) in r4".
the printed name of the a21r4 is "Room 4".
Understand "Area 21 in Room 4" as a21r4.
the parent of the a21r4 is r4.
the X of the a21r4 is "549529.8667450614".
the Y of the a21r4 is "2522115.7147068046".

[create a22r4]
a22r4 is a area. "An area (22) in r4".
the printed name of the a22r4 is "Room 4".
Understand "Area 22 in Room 4" as a22r4.
the parent of the a22r4 is r4.
the X of the a22r4 is "549671.3307578816".
the Y of the a22r4 is "2522111.7006285763".

[create a25r4]
a25r4 is a area. "An area (25) in r4".
the printed name of the a25r4 is "Room 4".
Understand "Area 25 in Room 4" as a25r4.
the parent of the a25r4 is r4.
the X of the a25r4 is "549476.6749245133".
the Y of the a25r4 is "2521993.1334230937".

[create a38r4]
a38r4 is a area. "An area (38) in r4".
the printed name of the a38r4 is "Room 4".
Understand "Area 38 in Room 4" as a38r4.
the parent of the a38r4 is r4.
the X of the a38r4 is "549594.0".
the Y of the a38r4 is "2522009.0".

[create a39r4]
a39r4 is a area. "An area (39) in r4".
the printed name of the a39r4 is "Room 4".
Understand "Area 39 in Room 4" as a39r4.
the parent of the a39r4 is r4.
the X of the a39r4 is "549583.8896296298".
the Y of the a39r4 is "2521972.757777778".

[create a42r4]
a42r4 is a area. "An area (42) in r4".
the printed name of the a42r4 is "Room 4".
Understand "Area 42 in Room 4" as a42r4.
the parent of the a42r4 is r4.
the X of the a42r4 is "549758.0".
the Y of the a42r4 is "2522240.0".

[create a43r4]
a43r4 is a area. "An area (43) in r4".
the printed name of the a43r4 is "Room 4".
Understand "Area 43 in Room 4" as a43r4.
the parent of the a43r4 is r4.
the X of the a43r4 is "549718.7500000005".
the Y of the a43r4 is "2522182.2685051844".

[create a44r4]
a44r4 is a area. "An area (44) in r4".
the printed name of the a44r4 is "Room 4".
Understand "Area 44 in Room 4" as a44r4.
the parent of the a44r4 is r4.
the X of the a44r4 is "549444.0".
the Y of the a44r4 is "2522240.0".

[create a45r4]
a45r4 is a area. "An area (45) in r4".
the printed name of the a45r4 is "Room 4".
Understand "Area 45 in Room 4" as a45r4.
the parent of the a45r4 is r4.
the X of the a45r4 is "549483.2499999998".
the Y of the a45r4 is "2522183.8322758973".

[create a5r5]
a5r5 is a area. "An area (5) in r5".
the printed name of the a5r5 is "Room 5".
Understand "Area 5 in Room 5" as a5r5.
the parent of the a5r5 is r5.
the X of the a5r5 is "545290.2994279691".
the Y of the a5r5 is "2520142.112798646".

[create a6r5]
a6r5 is a area. "An area (6) in r5".
the printed name of the a6r5 is "Room 5".
Understand "Area 6 in Room 5" as a6r5.
the parent of the a6r5 is r5.
the X of the a6r5 is "546033.8207015462".
the Y of the a6r5 is "2521110.4533167714".

[create a9r5]
a9r5 is a area. "An area (9) in r5".
the printed name of the a9r5 is "Room 5".
Understand "Area 9 in Room 5" as a9r5.
the parent of the a9r5 is r5.
the X of the a9r5 is "546775.3062686108".
the Y of the a9r5 is "2520203.255702281".

[create a10r5]
a10r5 is a area. "An area (10) in r5".
the printed name of the a10r5 is "Room 5".
Understand "Area 10 in Room 5" as a10r5.
the parent of the a10r5 is r5.
the X of the a10r5 is "546816.8343503118".
the Y of the a10r5 is "2520209.772325996".

[create a14r5]
a14r5 is a area. "An area (14) in r5".
the printed name of the a14r5 is "Room 5".
Understand "Area 14 in Room 5" as a14r5.
the parent of the a14r5 is r5.
the X of the a14r5 is "547027.5".
the Y of the a14r5 is "2520003.5534213684".

[create a15r5]
a15r5 is a area. "An area (15) in r5".
the printed name of the a15r5 is "Room 5".
Understand "Area 15 in Room 5" as a15r5.
the parent of the a15r5 is r5.
the X of the a15r5 is "545048.8999999999".
the Y of the a15r5 is "2520021.5829838095".

[create a34r5]
a34r5 is a area. "An area (34) in r5".
the printed name of the a34r5 is "Room 5".
Understand "Area 34 in Room 5" as a34r5.
the parent of the a34r5 is r5.
the X of the a34r5 is "545363.0".
the Y of the a34r5 is "2520000.0".

[create a35r5]
a35r5 is a area. "An area (35) in r5".
the printed name of the a35r5 is "Room 5".
Understand "Area 35 in Room 5" as a35r5.
the parent of the a35r5 is r5.
the X of the a35r5 is "545328.0999999999".
the Y of the a35r5 is "2520073.628215993".

[create a36r5]
a36r5 is a area. "An area (36) in r5".
the printed name of the a36r5 is "Room 5".
Understand "Area 36 in Room 5" as a36r5.
the parent of the a36r5 is r5.
the X of the a36r5 is "546744.0".
the Y of the a36r5 is "2520000.0".

[create a37r5]
a37r5 is a area. "An area (37) in r5".
the printed name of the a37r5 is "Room 5".
Understand "Area 37 in Room 5" as a37r5.
the parent of the a37r5 is r5.
the X of the a37r5 is "546707.0".
the Y of the a37r5 is "2520071.3419402256".

[create a54r5]
a54r5 is a area. "An area (54) in r5".
the printed name of the a54r5 is "Room 5".
Understand "Area 54 in Room 5" as a54r5.
the parent of the a54r5 is r5.
the X of the a54r5 is "547030.0".
the Y of the a54r5 is "2520000.0".

[create a55r5]
a55r5 is a area. "An area (55) in r5".
the printed name of the a55r5 is "Room 5".
Understand "Area 55 in Room 5" as a55r5.
the parent of the a55r5 is r5.
the X of the a55r5 is "545014.0".
the Y of the a55r5 is "2520000.0".

[create d0]
d0 is a door. "A door between a11r0 and a10r5".
d0 is south of a11r0 and north of a10r5.

[create d1]
d1 is a door. "A door between a33r0 and a6r5".
d1 is south of a33r0 and north of a6r5.

[create d2]
d2 is a door. "A door between a2r1 and a3r2".
d2 is north of a2r1 and south of a3r2.

[create d4]
d4 is a door. "A door between a7r0 and a8r1".
d4 is south of a7r0 and north of a8r1.

[create d8]
d8 is a door. "A door between a23r2 and a13r3".
d8 is south of a23r2 and north of a13r3.

[create d10]
d10 is a door. "A door between a32r3 and a25r4".
d10 is north of a32r3 and south of a25r4.

[create d11]
d11 is a door. "A door between a4r3 and a22r4".
d11 is north of a4r3 and south of a22r4.

northeast of a0r3 is southwest of a1r3.

northwest of a1r3 is southeast of a4r3.

northeast of a5r5 is southwest of a6r5.

southeast of a6r5 is northwest of a9r5.

east of a9r5 is west of a10r5.

north of a12r3 is south of a13r3.

southeast of a10r5 is northwest of a14r5.

southwest of a5r5 is northeast of a15r5.

southwest of a12r3 is northeast of a16r3.

southeast of a1r3 is northwest of a17r3.

southwest of a8r1 is northeast of a18r1.

southeast of a2r1 is northwest of a19r1.

southeast of a12r3 is northwest of a20r3.

east of a21r4 is west of a22r4.

northeast of a23r2 is southwest of a24r2.

southwest of a21r4 is northeast of a25r4.

southeast of a26r0 is northwest of a27r0.

southeast of a28r0 is northwest of a29r0.

northeast of a7r0 is southwest of a30r0.

west of a7r0 is east of a11r0.

west of a11r0 is east of a29r0.

east of a3r2 is west of a23r2.

northwest of a3r2 is southeast of a31r2.

northeast of a13r3 is southwest of a32r3.

west of a4r3 is east of a32r3.

southeast of a27r0 is northwest of a33r0.

southwest of a29r0 is northeast of a33r0.

northwest of a34r5 is southeast of a35r5.

northwest of a36r5 is southeast of a37r5.

south of a38r4 is north of a39r4.

southwest of a40r0 is northeast of a41r0.

southwest of a42r4 is northeast of a43r4.

southeast of a44r4 is northwest of a45r4.

southwest of a18r1 is northeast of a46r1.

southeast of a19r1 is northwest of a47r1.

southeast of a20r3 is northwest of a48r3.

southwest of a0r3 is northeast of a49r3.

northwest of a26r0 is southeast of a50r0.

north of a28r0 is south of a51r0.

northeast of a30r0 is southwest of a52r0.

northwest of a31r2 is southeast of a53r2.

southeast of a14r5 is northwest of a54r5.

southwest of a15r5 is northeast of a55r5.

southeast of a16r3 is northwest of a56r3.

southeast of a17r3 is northwest of a57r3.

northeast of a24r2 is southwest of a58r2.

southwest of a9r5 is northeast of a37r5.

east of a25r4 is west of a39r4.

northeast of a27r0 is southwest of a41r0.

northeast of a22r4 is southwest of a43r4.

northwest of a21r4 is southeast of a45r4.

southeast of a5r5 is northwest of a35r5.

west of a2r1 is east of a8r1.

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
