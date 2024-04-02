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

[create a2r0]
a2r0 is a area. "An area (2) in r0".
the printed name of the a2r0 is "Room 0".
Understand "Area 2 in Room 0" as a2r0.
the parent of the a2r0 is r0.
the X of the a2r0 is "545397.9893489797".
the Y of the a2r0 is "1948279.158965812".

[create a4r0]
a4r0 is a area. "An area (4) in r0".
the printed name of the a4r0 is "Room 0".
Understand "Area 4 in Room 0" as a4r0.
the parent of the a4r0 is r0.
the X of the a4r0 is "547160.4400065946".
the Y of the a4r0 is "1948114.8467057191".

[create a7r0]
a7r0 is a area. "An area (7) in r0".
the printed name of the a7r0 is "Room 0".
Understand "Area 7 in Room 0" as a7r0.
the parent of the a7r0 is r0.
the X of the a7r0 is "546818.8701322983".
the Y of the a7r0 is "1948006.3385332255".

[create a27r0]
a27r0 is a area. "An area (27) in r0".
the printed name of the a27r0 is "Room 0".
Understand "Area 27 in Room 0" as a27r0.
the parent of the a27r0 is r0.
the X of the a27r0 is "546787.5".
the Y of the a27r0 is "1948401.8390804601".

[create a28r0]
a28r0 is a area. "An area (28) in r0".
the printed name of the a28r0 is "Room 0".
Understand "Area 28 in Room 0" as a28r0.
the parent of the a28r0 is r0.
the X of the a28r0 is "546930.0254391992".
the Y of the a28r0 is "1948262.2673831559".

[create a29r0]
a29r0 is a area. "An area (29) in r0".
the printed name of the a29r0 is "Room 0".
Understand "Area 29 in Room 0" as a29r0.
the parent of the a29r0 is r0.
the X of the a29r0 is "545274.8999999999".
the Y of the a29r0 is "1948401.3793103448".

[create a34r0]
a34r0 is a area. "An area (34) in r0".
the printed name of the a34r0 is "Room 0".
Understand "Area 34 in Room 0" as a34r0.
the parent of the a34r0 is r0.
the X of the a34r0 is "546507.5179617628".
the Y of the a34r0 is "1948234.8403829157".

[create a35r0]
a35r0 is a area. "An area (35) in r0".
the printed name of the a35r0 is "Room 0".
Understand "Area 35 in Room 0" as a35r0.
the parent of the a35r0 is r0.
the X of the a35r0 is "547070.6528639615".
the Y of the a35r0 is "1948275.3847660443".

[create a37r0]
a37r0 is a area. "An area (37) in r0".
the printed name of the a37r0 is "Room 0".
Understand "Area 37 in Room 0" as a37r0.
the parent of the a37r0 is r0.
the X of the a37r0 is "547152.4999999998".
the Y of the a37r0 is "1948388.0236277871".

[create a38r0]
a38r0 is a area. "An area (38) in r0".
the printed name of the a38r0 is "Room 0".
Understand "Area 38 in Room 0" as a38r0.
the parent of the a38r0 is r0.
the X of the a38r0 is "547032.0".
the Y of the a38r0 is "1948113.0".

[create a39r0]
a39r0 is a area. "An area (39) in r0".
the printed name of the a39r0 is "Room 0".
Understand "Area 39 in Room 0" as a39r0.
the parent of the a39r0 is r0.
the X of the a39r0 is "547049.9395632693".
the Y of the a39r0 is "1948078.1094961239".

[create a40r0]
a40r0 is a area. "An area (40) in r0".
the printed name of the a40r0 is "Room 0".
Understand "Area 40 in Room 0" as a40r0.
the parent of the a40r0 is r0.
the X of the a40r0 is "546575.0".
the Y of the a40r0 is "1948364.0".

[create a41r0]
a41r0 is a area. "An area (41) in r0".
the printed name of the a41r0 is "Room 0".
Understand "Area 41 in Room 0" as a41r0.
the parent of the a41r0 is r0.
the X of the a41r0 is "546515.4097654722".
the Y of the a41r0 is "1948250.6730077486".

[create a43r0]
a43r0 is a area. "An area (43) in r0".
the printed name of the a43r0 is "Room 0".
Understand "Area 43 in Room 0" as a43r0.
the parent of the a43r0 is r0.
the X of the a43r0 is "546751.0".
the Y of the a43r0 is "1948440.0".

[create a44r0]
a44r0 is a area. "An area (44) in r0".
the printed name of the a44r0 is "Room 0".
Understand "Area 44 in Room 0" as a44r0.
the parent of the a44r0 is r0.
the X of the a44r0 is "545237.0".
the Y of the a44r0 is "1948440.0".

[create a54r0]
a54r0 is a area. "An area (54) in r0".
the printed name of the a54r0 is "Room 0".
Understand "Area 54 in Room 0" as a54r0.
the parent of the a54r0 is r0.
the X of the a54r0 is "547189.0".
the Y of the a54r0 is "1948440.0".

[create a5r1]
a5r1 is a area. "An area (5) in r1".
the printed name of the a5r1 is "Room 1".
Understand "Area 5 in Room 1" as a5r1.
the parent of the a5r1 is r1.
the X of the a5r1 is "547779.624643379".
the Y of the a5r1 is "1946359.8817214698".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "547700.75".
the Y of the a14r1 is "1946251.7428626001".

[create a15r1]
a15r1 is a area. "An area (15) in r1".
the printed name of the a15r1 is "Room 1".
Understand "Area 15 in Room 1" as a15r1.
the parent of the a15r1 is r1.
the X of the a15r1 is "547820.192689911".
the Y of the a15r1 is "1946357.8341229067".

[create a16r1]
a16r1 is a area. "An area (16) in r1".
the printed name of the a16r1 is "Room 1".
Understand "Area 16 in Room 1" as a16r1.
the parent of the a16r1 is r1.
the X of the a16r1 is "547897.25".
the Y of the a16r1 is "1946251.4843251861".

[create a49r1]
a49r1 is a area. "An area (49) in r1".
the printed name of the a49r1 is "Room 1".
Understand "Area 49 in Room 1" as a49r1.
the parent of the a49r1 is r1.
the X of the a49r1 is "547668.0".
the Y of the a49r1 is "1946200.0".

[create a50r1]
a50r1 is a area. "An area (50) in r1".
the printed name of the a50r1 is "Room 1".
Understand "Area 50 in Room 1" as a50r1.
the parent of the a50r1 is r1.
the X of the a50r1 is "547930.0".
the Y of the a50r1 is "1946200.0".

[create a17r2]
a17r2 is a area. "An area (17) in r2".
the printed name of the a17r2 is "Room 2".
Understand "Area 17 in Room 2" as a17r2.
the parent of the a17r2 is r2.
the X of the a17r2 is "548586.051152792".
the Y of the a17r2 is "1948261.7482603411".

[create a20r2]
a20r2 is a area. "An area (20) in r2".
the printed name of the a20r2 is "Room 2".
Understand "Area 20 in Room 2" as a20r2.
the parent of the a20r2 is r2.
the X of the a20r2 is "548661.0934343643".
the Y of the a20r2 is "1948250.5944565693".

[create a26r2]
a26r2 is a area. "An area (26) in r2".
the printed name of the a26r2 is "Room 2".
Understand "Area 26 in Room 2" as a26r2.
the parent of the a26r2 is r2.
the X of the a26r2 is "548497.2000000002".
the Y of the a26r2 is "1948385.3789812021".

[create a31r2]
a31r2 is a area. "An area (31) in r2".
the printed name of the a31r2 is "Room 2".
Understand "Area 31 in Room 2" as a31r2.
the parent of the a31r2 is r2.
the X of the a31r2 is "548810.7999999998".
the Y of the a31r2 is "1948401.3793103448".

[create a42r2]
a42r2 is a area. "An area (42) in r2".
the printed name of the a42r2 is "Room 2".
Understand "Area 42 in Room 2" as a42r2.
the parent of the a42r2 is r2.
the X of the a42r2 is "548458.0".
the Y of the a42r2 is "1948440.0".

[create a45r2]
a45r2 is a area. "An area (45) in r2".
the printed name of the a45r2 is "Room 2".
Understand "Area 45 in Room 2" as a45r2.
the parent of the a45r2 is r2.
the X of the a45r2 is "548850.0".
the Y of the a45r2 is "1948440.0".

[create a19r3]
a19r3 is a area. "An area (19) in r3".
the printed name of the a19r3 is "Room 3".
Understand "Area 19 in Room 3" as a19r3.
the parent of the a19r3 is r3.
the X of the a19r3 is "548709.3794133378".
the Y of the a19r3 is "1946341.2424854166".

[create a21r3]
a21r3 is a area. "An area (21) in r3".
the printed name of the a21r3 is "Room 3".
Understand "Area 21 in Room 3" as a21r3.
the parent of the a21r3 is r3.
the X of the a21r3 is "548604.125".
the Y of the a21r3 is "1946239.2083333335".

[create a22r3]
a22r3 is a area. "An area (22) in r3".
the printed name of the a22r3 is "Room 3".
Understand "Area 22 in Room 3" as a22r3.
the parent of the a22r3 is r3.
the X of the a22r3 is "548814.875".
the Y of the a22r3 is "1946238.6206896554".

[create a55r3]
a55r3 is a area. "An area (55) in r3".
the printed name of the a55r3 is "Room 3".
Understand "Area 55 in Room 3" as a55r3.
the parent of the a55r3 is r3.
the X of the a55r3 is "548569.0".
the Y of the a55r3 is "1946200.0".

[create a56r3]
a56r3 is a area. "An area (56) in r3".
the printed name of the a56r3 is "Room 3".
Understand "Area 56 in Room 3" as a56r3.
the parent of the a56r3 is r3.
the X of the a56r3 is "548850.0".
the Y of the a56r3 is "1946200.0".

[create a3r4]
a3r4 is a area. "An area (3) in r4".
the printed name of the a3r4 is "Room 4".
Understand "Area 3 in Room 4" as a3r4.
the parent of the a3r4 is r4.
the X of the a3r4 is "545398.1496501891".
the Y of the a3r4 is "1946360.3815294644".

[create a6r4]
a6r4 is a area. "An area (6) in r4".
the printed name of the a6r4 is "Room 4".
Understand "Area 6 in Room 4" as a6r4.
the parent of the a6r4 is r4.
the X of the a6r4 is "546823.1122433166".
the Y of the a6r4 is "1946619.0413653776".

[create a18r4]
a18r4 is a area. "An area (18) in r4".
the printed name of the a18r4 is "Room 4".
Understand "Area 18 in Room 4" as a18r4.
the parent of the a18r4 is r4.
the X of the a18r4 is "546891.4389885867".
the Y of the a18r4 is "1946340.640612795".

[create a23r4]
a23r4 is a area. "An area (23) in r4".
the printed name of the a23r4 is "Room 4".
Understand "Area 23 in Room 4" as a23r4.
the parent of the a23r4 is r4.
the X of the a23r4 is "546786.125".
the Y of the a23r4 is "1946237.0422427035".

[create a24r4]
a24r4 is a area. "An area (24) in r4".
the printed name of the a24r4 is "Room 4".
Understand "Area 24 in Room 4" as a24r4.
the parent of the a24r4 is r4.
the X of the a24r4 is "546996.875".
the Y of the a24r4 is "1946236.343045843".

[create a25r4]
a25r4 is a area. "An area (25) in r4".
the printed name of the a25r4 is "Room 4".
Understand "Area 25 in Room 4" as a25r4.
the parent of the a25r4 is r4.
the X of the a25r4 is "545273.090909091".
the Y of the a25r4 is "1946238.6206896554".

[create a57r4]
a57r4 is a area. "An area (57) in r4".
the printed name of the a57r4 is "Room 4".
Understand "Area 57 in Room 4" as a57r4.
the parent of the a57r4 is r4.
the X of the a57r4 is "546751.0".
the Y of the a57r4 is "1946200.0".

[create a58r4]
a58r4 is a area. "An area (58) in r4".
the printed name of the a58r4 is "Room 4".
Understand "Area 58 in Room 4" as a58r4.
the parent of the a58r4 is r4.
the X of the a58r4 is "547032.0".
the Y of the a58r4 is "1946200.0".

[create a59r4]
a59r4 is a area. "An area (59) in r4".
the printed name of the a59r4 is "Room 4".
Understand "Area 59 in Room 4" as a59r4.
the parent of the a59r4 is r4.
the X of the a59r4 is "545237.0".
the Y of the a59r4 is "1946200.0".

[create a0r5]
a0r5 is a area. "An area (0) in r5".
the printed name of the a0r5 is "Room 5".
Understand "Area 0 in Room 5" as a0r5.
the parent of the a0r5 is r5.
the X of the a0r5 is "549036.575887403".
the Y of the a0r5 is "1947352.0".

[create a8r5]
a8r5 is a area. "An area (8) in r5".
the printed name of the a8r5 is "Room 5".
Understand "Area 8 in Room 5" as a8r5.
the parent of the a8r5 is r5.
the X of the a8r5 is "550056.7198228128".
the Y of the a8r5 is "1947352.0".

[create a10r5]
a10r5 is a area. "An area (10) in r5".
the printed name of the a10r5 is "Room 5".
Understand "Area 10 in Room 5" as a10r5.
the parent of the a10r5 is r5.
the X of the a10r5 is "550149.1428571432".
the Y of the a10r5 is "1947448.0".

[create a11r5]
a11r5 is a area. "An area (11) in r5".
the printed name of the a11r5 is "Room 5".
Understand "Area 11 in Room 5" as a11r5.
the parent of the a11r5 is r5.
the X of the a11r5 is "549023.934825828".
the Y of the a11r5 is "1948279.0026747533".

[create a30r5]
a30r5 is a area. "An area (30) in r5".
the printed name of the a30r5 is "Room 5".
Understand "Area 30 in Room 5" as a30r5.
the parent of the a30r5 is r5.
the X of the a30r5 is "548902.0526315789".
the Y of the a30r5 is "1948401.3793103448".

[create a32r5]
a32r5 is a area. "An area (32) in r5".
the printed name of the a32r5 is "Room 5".
Understand "Area 32 in Room 5" as a32r5.
the parent of the a32r5 is r5.
the X of the a32r5 is "550216.9660090085".
the Y of the a32r5 is "1948312.6264538604".

[create a33r5]
a33r5 is a area. "An area (33) in r5".
the printed name of the a33r5 is "Room 5".
Understand "Area 33 in Room 5" as a33r5.
the parent of the a33r5 is r5.
the X of the a33r5 is "550307.9473684211".
the Y of the a33r5 is "1948408.0".

[create a36r5]
a36r5 is a area. "An area (36) in r5".
the printed name of the a36r5 is "Room 5".
Understand "Area 36 in Room 5" as a36r5.
the parent of the a36r5 is r5.
the X of the a36r5 is "550309.71875".
the Y of the a36r5 is "1948216.0".

[create a46r5]
a46r5 is a area. "An area (46) in r5".
the printed name of the a46r5 is "Room 5".
Understand "Area 46 in Room 5" as a46r5.
the parent of the a46r5 is r5.
the X of the a46r5 is "550347.0".
the Y of the a46r5 is "1948440.0".

[create a52r5]
a52r5 is a area. "An area (52) in r5".
the printed name of the a52r5 is "Room 5".
Understand "Area 52 in Room 5" as a52r5.
the parent of the a52r5 is r5.
the X of the a52r5 is "550186.0".
the Y of the a52r5 is "1947480.0".

[create a53r5]
a53r5 is a area. "An area (53) in r5".
the printed name of the a53r5 is "Room 5".
Understand "Area 53 in Room 5" as a53r5.
the parent of the a53r5 is r5.
the X of the a53r5 is "550347.0".
the Y of the a53r5 is "1948184.0".

[create a1r6]
a1r6 is a area. "An area (1) in r6".
the printed name of the a1r6 is "Room 6".
Understand "Area 1 in Room 6" as a1r6.
the parent of the a1r6 is r6.
the X of the a1r6 is "549008.606572343".
the Y of the a1r6 is "1946345.861537123".

[create a9r6]
a9r6 is a area. "An area (9) in r6".
the printed name of the a9r6 is "Room 6".
Understand "Area 9 in Room 6" as a9r6.
the parent of the a9r6 is r6.
the X of the a9r6 is "550149.1428571428".
the Y of the a9r6 is "1947256.0".

[create a12r6]
a12r6 is a area. "An area (12) in r6".
the printed name of the a12r6 is "Room 6".
Understand "Area 12 in Room 6" as a12r6.
the parent of the a12r6 is r6.
the X of the a12r6 is "549117.625".
the Y of the a12r6 is "1946239.3846153845".

[create a13r6]
a13r6 is a area. "An area (13) in r6".
the printed name of the a13r6 is "Room 6".
Understand "Area 13 in Room 6" as a13r6.
the parent of the a13r6 is r6.
the X of the a13r6 is "548899.375".
the Y of the a13r6 is "1946238.6206896554".

[create a47r6]
a47r6 is a area. "An area (47) in r6".
the printed name of the a47r6 is "Room 6".
Understand "Area 47 in Room 6" as a47r6.
the parent of the a47r6 is r6.
the X of the a47r6 is "549154.0".
the Y of the a47r6 is "1946200.0".

[create a48r6]
a48r6 is a area. "An area (48) in r6".
the printed name of the a48r6 is "Room 6".
Understand "Area 48 in Room 6" as a48r6.
the parent of the a48r6 is r6.
the X of the a48r6 is "548863.0".
the Y of the a48r6 is "1946200.0".

[create a51r6]
a51r6 is a area. "An area (51) in r6".
the printed name of the a51r6 is "Room 6".
Understand "Area 51 in Room 6" as a51r6.
the parent of the a51r6 is r6.
the X of the a51r6 is "550186.0".
the Y of the a51r6 is "1947224.0".

[create d0]
d0 is a door. "A door between a15r1 and a17r2".
d0 is north of a15r1 and south of a17r2.

[create d2]
d2 is a door. "A door between a20r2 and a19r3".
d2 is south of a20r2 and north of a19r3.

[create d4]
d4 is a door. "A door between a0r5 and a1r6".
d4 is south of a0r5 and north of a1r6.

[create d5]
d5 is a door. "A door between a8r5 and a9r6".
d5 is southeast of a8r5 and northwest of a9r6.

[create d6]
d6 is a door. "A door between a2r0 and a3r4".
d6 is south of a2r0 and north of a3r4.

[create d7]
d7 is a door. "A door between a7r0 and a6r4".
d7 is south of a7r0 and north of a6r4.

[create d8]
d8 is a door. "A door between a4r0 and a5r1".
d8 is south of a4r0 and north of a5r1.

east of a0r5 is west of a8r5.

northeast of a8r5 is southwest of a10r5.

north of a0r5 is south of a11r5.

southeast of a1r6 is northwest of a12r6.

southwest of a1r6 is northeast of a13r6.

east of a3r4 is west of a6r4.

southwest of a5r1 is northeast of a14r1.

southeast of a15r1 is northwest of a16r1.

south of a6r4 is north of a18r4.

southwest of a19r3 is northeast of a21r3.

southeast of a19r3 is northwest of a22r3.

southwest of a18r4 is northeast of a23r4.

southeast of a18r4 is northwest of a24r4.

southwest of a3r4 is northeast of a25r4.

east of a17r2 is west of a20r2.

northwest of a17r2 is southeast of a26r2.

southeast of a27r0 is northwest of a28r0.

northwest of a2r0 is southeast of a29r0.

northwest of a11r5 is southeast of a30r5.

northeast of a20r2 is southwest of a31r2.

northeast of a32r5 is southwest of a33r5.

northwest of a7r0 is southeast of a34r0.

northeast of a7r0 is southwest of a28r0.

northwest of a4r0 is southeast of a35r0.

southeast of a32r5 is northwest of a36r5.

east of a2r0 is west of a34r0.

east of a11r5 is west of a32r5.

northeast of a35r0 is southwest of a37r0.

east of a28r0 is west of a35r0.

southeast of a38r0 is northwest of a39r0.

southwest of a40r0 is northeast of a41r0.

northwest of a26r2 is southeast of a42r2.

northwest of a27r0 is southeast of a43r0.

northwest of a29r0 is southeast of a44r0.

northeast of a31r2 is southwest of a45r2.

northeast of a33r5 is southwest of a46r5.

southeast of a12r6 is northwest of a47r6.

southwest of a13r6 is northeast of a48r6.

southwest of a14r1 is northeast of a49r1.

southeast of a16r1 is northwest of a50r1.

southeast of a9r6 is northwest of a51r6.

northeast of a10r5 is southwest of a52r5.

southeast of a36r5 is northwest of a53r5.

northeast of a37r0 is southwest of a54r0.

southwest of a21r3 is northeast of a55r3.

southeast of a22r3 is northwest of a56r3.

southwest of a23r4 is northeast of a57r4.

southeast of a24r4 is northwest of a58r4.

southwest of a25r4 is northeast of a59r4.

east of a5r1 is west of a15r1.

west of a4r0 is east of a39r0.

northeast of a34r0 is southwest of a41r0.

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
the player is in a49r1.

the orientation of the player is 6.
