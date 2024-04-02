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

[create a4r0]
a4r0 is a area. "An area (4) in r0".
the printed name of the a4r0 is "Room 0".
Understand "Area 4 in Room 0" as a4r0.
the parent of the a4r0 is r0.
the X of the a4r0 is "545893.375".
the Y of the a4r0 is "2550436.923076923".

[create a5r0]
a5r0 is a area. "An area (5) in r0".
the printed name of the a5r0 is "Room 0".
Understand "Area 5 in Room 0" as a5r0.
the parent of the a5r0 is r0.
the X of the a5r0 is "546002.5000000002".
the Y of the a5r0 is "2550545.516271034".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "546111.625".
the Y of the a6r0 is "2550436.923076923".

[create a27r0]
a27r0 is a area. "An area (27) in r0".
the printed name of the a27r0 is "Room 0".
Understand "Area 27 in Room 0" as a27r0.
the parent of the a27r0 is r0.
the X of the a27r0 is "546002.5".
the Y of the a27r0 is "2551447.25".

[create a56r0]
a56r0 is a area. "An area (56) in r0".
the printed name of the a56r0 is "Room 0".
Understand "Area 56 in Room 0" as a56r0.
the parent of the a56r0 is r0.
the X of the a56r0 is "545857.0".
the Y of the a56r0 is "2550400.0".

[create a57r0]
a57r0 is a area. "An area (57) in r0".
the printed name of the a57r0 is "Room 0".
Understand "Area 57 in Room 0" as a57r0.
the parent of the a57r0 is r0.
the X of the a57r0 is "546148.0".
the Y of the a57r0 is "2550400.0".

[create a9r1]
a9r1 is a area. "An area (9) in r1".
the printed name of the a9r1 is "Room 1".
Understand "Area 9 in Room 1" as a9r1.
the parent of the a9r1 is r1.
the X of the a9r1 is "547981.7499999998".
the Y of the a9r1 is "2552620.4929351737".

[create a10r1]
a10r1 is a area. "An area (10) in r1".
the printed name of the a10r1 is "Room 1".
Understand "Area 10 in Room 1" as a10r1.
the parent of the a10r1 is r1.
the X of the a10r1 is "548128.5078295618".
the Y of the a10r1 is "2552481.6406066753".

[create a13r1]
a13r1 is a area. "An area (13) in r1".
the printed name of the a13r1 is "Room 1".
Understand "Area 13 in Room 1" as a13r1.
the parent of the a13r1 is r1.
the X of the a13r1 is "545044.3999999999".
the Y of the a13r1 is "2552617.4075".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "545282.5862396206".
the Y of the a14r1 is "2552491.0611061687".

[create a15r1]
a15r1 is a area. "An area (15) in r1".
the printed name of the a15r1 is "Room 1".
Understand "Area 15 in Room 1" as a15r1.
the parent of the a15r1 is r1.
the X of the a15r1 is "546545.1666666669".
the Y of the a15r1 is "2552613.599623628".

[create a16r1]
a16r1 is a area. "An area (16) in r1".
the printed name of the a16r1 is "Room 1".
Understand "Area 16 in Room 1" as a16r1.
the parent of the a16r1 is r1.
the X of the a16r1 is "546776.0984795617".
the Y of the a16r1 is "2552461.1913606157".

[create a17r1]
a17r1 is a area. "An area (17) in r1".
the printed name of the a17r1 is "Room 1".
Understand "Area 17 in Room 1" as a17r1.
the parent of the a17r1 is r1.
the X of the a17r1 is "546998.1666666663".
the Y of the a17r1 is "2552615.9142567255".

[create a18r1]
a18r1 is a area. "An area (18) in r1".
the printed name of the a18r1 is "Room 1".
Understand "Area 18 in Room 1" as a18r1.
the parent of the a18r1 is r1.
the X of the a18r1 is "548343.5".
the Y of the a18r1 is "2552614.3790818145".

[create a24r1]
a24r1 is a area. "An area (24) in r1".
the printed name of the a24r1 is "Room 1".
Understand "Area 24 in Room 1" as a24r1.
the parent of the a24r1 is r1.
the X of the a24r1 is "548113.4604242069".
the Y of the a24r1 is "2552405.0055419225".

[create a28r1]
a28r1 is a area. "An area (28) in r1".
the printed name of the a28r1 is "Room 1".
Understand "Area 28 in Room 1" as a28r1.
the parent of the a28r1 is r1.
the X of the a28r1 is "546776.2269591065".
the Y of the a28r1 is "2552455.8521109256".

[create a29r1]
a29r1 is a area. "An area (29) in r1".
the printed name of the a29r1 is "Room 1".
Understand "Area 29 in Room 1" as a29r1.
the parent of the a29r1 is r1.
the X of the a29r1 is "547430.6855935078".
the Y of the a29r1 is "2551630.6292565824".

[create a39r1]
a39r1 is a area. "An area (39) in r1".
the printed name of the a39r1 is "Room 1".
Understand "Area 39 in Room 1" as a39r1.
the parent of the a39r1 is r1.
the X of the a39r1 is "545363.0".
the Y of the a39r1 is "2552640.0".

[create a40r1]
a40r1 is a area. "An area (40) in r1".
the printed name of the a40r1 is "Room 1".
Understand "Area 40 in Room 1" as a40r1.
the parent of the a40r1 is r1.
the X of the a40r1 is "545327.6000000003".
the Y of the a40r1 is "2552569.4234756106".

[create a44r1]
a44r1 is a area. "An area (44) in r1".
the printed name of the a44r1 is "Room 1".
Understand "Area 44 in Room 1" as a44r1.
the parent of the a44r1 is r1.
the X of the a44r1 is "547957.0".
the Y of the a44r1 is "2552640.0".

[create a50r1]
a50r1 is a area. "An area (50) in r1".
the printed name of the a50r1 is "Room 1".
Understand "Area 50 in Room 1" as a50r1.
the parent of the a50r1 is r1.
the X of the a50r1 is "545009.0".
the Y of the a50r1 is "2552640.0".

[create a51r1]
a51r1 is a area. "An area (51) in r1".
the printed name of the a51r1 is "Room 1".
Understand "Area 51 in Room 1" as a51r1.
the parent of the a51r1 is r1.
the X of the a51r1 is "546517.0".
the Y of the a51r1 is "2552640.0".

[create a52r1]
a52r1 is a area. "An area (52) in r1".
the printed name of the a52r1 is "Room 1".
Understand "Area 52 in Room 1" as a52r1.
the parent of the a52r1 is r1.
the X of the a52r1 is "547025.0".
the Y of the a52r1 is "2552640.0".

[create a53r1]
a53r1 is a area. "An area (53) in r1".
the printed name of the a53r1 is "Room 1".
Understand "Area 53 in Room 1" as a53r1.
the parent of the a53r1 is r1.
the X of the a53r1 is "548373.0".
the Y of the a53r1 is "2552640.0".

[create a25r2]
a25r2 is a area. "An area (25) in r2".
the printed name of the a25r2 is "Room 2".
Understand "Area 25 in Room 2" as a25r2.
the parent of the a25r2 is r2.
the X of the a25r2 is "548657.1566116107".
the Y of the a25r2 is "2550553.4894037787".

[create a26r2]
a26r2 is a area. "An area (26) in r2".
the printed name of the a26r2 is "Room 2".
Understand "Area 26 in Room 2" as a26r2.
the parent of the a26r2 is r2.
the X of the a26r2 is "548719.9412801261".
the Y of the a26r2 is "2550554.610558574".

[create a32r2]
a32r2 is a area. "An area (32) in r2".
the printed name of the a32r2 is "Room 2".
Understand "Area 32 in Room 2" as a32r2.
the parent of the a32r2 is r2.
the X of the a32r2 is "548577.125".
the Y of the a32r2 is "2550450.164502177".

[create a33r2]
a33r2 is a area. "An area (33) in r2".
the printed name of the a33r2 is "Room 2".
Understand "Area 33 in Room 2" as a33r2.
the parent of the a33r2 is r2.
the X of the a33r2 is "548799.875".
the Y of the a33r2 is "2550450.099474627".

[create a48r2]
a48r2 is a area. "An area (48) in r2".
the printed name of the a48r2 is "Room 2".
Understand "Area 48 in Room 2" as a48r2.
the parent of the a48r2 is r2.
the X of the a48r2 is "548540.0".
the Y of the a48r2 is "2550400.0".

[create a49r2]
a49r2 is a area. "An area (49) in r2".
the printed name of the a49r2 is "Room 2".
Understand "Area 49 in Room 2" as a49r2.
the parent of the a49r2 is r2.
the X of the a49r2 is "548837.0".
the Y of the a49r2 is "2550400.0".

[create a19r3]
a19r3 is a area. "An area (19) in r3".
the printed name of the a19r3 is "Room 3".
Understand "Area 19 in Room 3" as a19r3.
the parent of the a19r3 is r3.
the X of the a19r3 is "549442.8305982075".
the Y of the a19r3 is "2552498.8947038357".

[create a20r3]
a20r3 is a area. "An area (20) in r3".
the printed name of the a20r3 is "Room 3".
Understand "Area 20 in Room 3" as a20r3.
the parent of the a20r3 is r3.
the X of the a20r3 is "549515.9000000001".
the Y of the a20r3 is "2552589.466754617".

[create a21r3]
a21r3 is a area. "An area (21) in r3".
the printed name of the a21r3 is "Room 3".
Understand "Area 21 in Room 3" as a21r3.
the parent of the a21r3 is r3.
the X of the a21r3 is "549235.0999999997".
the Y of the a21r3 is "2552589.5239782464".

[create a22r3]
a22r3 is a area. "An area (22) in r3".
the printed name of the a22r3 is "Room 3".
Understand "Area 22 in Room 3" as a22r3.
the parent of the a22r3 is r3.
the X of the a22r3 is "549306.5565103933".
the Y of the a22r3 is "2552502.537885611".

[create a41r3]
a41r3 is a area. "An area (41) in r3".
the printed name of the a41r3 is "Room 3".
Understand "Area 41 in Room 3" as a41r3.
the parent of the a41r3 is r3.
the X of the a41r3 is "549551.0".
the Y of the a41r3 is "2552640.0".

[create a42r3]
a42r3 is a area. "An area (42) in r3".
the printed name of the a42r3 is "Room 3".
Understand "Area 42 in Room 3" as a42r3.
the parent of the a42r3 is r3.
the X of the a42r3 is "549200.0".
the Y of the a42r3 is "2552640.0".

[create a7r4]
a7r4 is a area. "An area (7) in r4".
the printed name of the a7r4 is "Room 4".
Understand "Area 7 in Room 4" as a7r4.
the parent of the a7r4 is r4.
the X of the a7r4 is "550065.0751328211".
the Y of the a7r4 is "2550554.3386380887".

[create a8r4]
a8r4 is a area. "An area (8) in r4".
the printed name of the a8r4 is "Room 4".
Understand "Area 8 in Room 4" as a8r4.
the parent of the a8r4 is r4.
the X of the a8r4 is "550146.0".
the Y of the a8r4 is "2550449.46063244".

[create a30r4]
a30r4 is a area. "An area (30) in r4".
the printed name of the a30r4 is "Room 4".
Understand "Area 30 in Room 4" as a30r4.
the parent of the a30r4 is r4.
the X of the a30r4 is "549936.0".
the Y of the a30r4 is "2550449.422623434".

[create a31r4]
a31r4 is a area. "An area (31) in r4".
the printed name of the a31r4 is "Room 4".
Understand "Area 31 in Room 4" as a31r4.
the parent of the a31r4 is r4.
the X of the a31r4 is "550016.2576810799".
the Y of the a31r4 is "2550553.704645209".

[create a46r4]
a46r4 is a area. "An area (46) in r4".
the printed name of the a46r4 is "Room 4".
Understand "Area 46 in Room 4" as a46r4.
the parent of the a46r4 is r4.
the X of the a46r4 is "550181.0".
the Y of the a46r4 is "2550400.0".

[create a47r4]
a47r4 is a area. "An area (47) in r4".
the printed name of the a47r4 is "Room 4".
Understand "Area 47 in Room 4" as a47r4.
the parent of the a47r4 is r4.
the X of the a47r4 is "549901.0".
the Y of the a47r4 is "2550400.0".

[create a11r5]
a11r5 is a area. "An area (11) in r5".
the printed name of the a11r5 is "Room 5".
Understand "Area 11 in Room 5" as a11r5.
the parent of the a11r5 is r5.
the X of the a11r5 is "550602.0274643078".
the Y of the a11r5 is "2552502.368728805".

[create a12r5]
a12r5 is a area. "An area (12) in r5".
the printed name of the a12r5 is "Room 5".
Understand "Area 12 in Room 5" as a12r5.
the parent of the a12r5 is r5.
the X of the a12r5 is "550747.4999999998".
the Y of the a12r5 is "2552609.7869122024".

[create a23r5]
a23r5 is a area. "An area (23) in r5".
the printed name of the a23r5 is "Room 5".
Understand "Area 23 in Room 5" as a23r5.
the parent of the a23r5 is r5.
the X of the a23r5 is "550528.4999999995".
the Y of the a23r5 is "2552591.129062261".

[create a43r5]
a43r5 is a area. "An area (43) in r5".
the printed name of the a43r5 is "Room 5".
Understand "Area 43 in Room 5" as a43r5.
the parent of the a43r5 is r5.
the X of the a43r5 is "550492.0".
the Y of the a43r5 is "2552640.0".

[create a45r5]
a45r5 is a area. "An area (45) in r5".
the printed name of the a45r5 is "Room 5".
Understand "Area 45 in Room 5" as a45r5.
the parent of the a45r5 is r5.
the X of the a45r5 is "550784.0".
the Y of the a45r5 is "2552640.0".

[create a0r6]
a0r6 is a area. "An area (0) in r6".
the printed name of the a0r6 is "Room 6".
Understand "Area 0 in Room 6" as a0r6.
the parent of the a0r6 is r6.
the X of the a0r6 is "546451.8999999999".
the Y of the a0r6 is "2550421.5829838095".

[create a1r6]
a1r6 is a area. "An area (1) in r6".
the printed name of the a1r6 is "Room 6".
Understand "Area 1 in Room 6" as a1r6.
the parent of the a1r6 is r6.
the X of the a1r6 is "546693.2994279691".
the Y of the a1r6 is "2550542.112798646".

[create a2r6]
a2r6 is a area. "An area (2) in r6".
the printed name of the a2r6 is "Room 6".
Understand "Area 2 in Room 6" as a2r6.
the parent of the a2r6 is r6.
the X of the a2r6 is "548149.7281516471".
the Y of the a2r6 is "2550547.941871377".

[create a3r6]
a3r6 is a area. "An area (3) in r6".
the printed name of the a3r6 is "Room 6".
Understand "Area 3 in Room 6" as a3r6.
the parent of the a3r6 is r6.
the X of the a3r6 is "548397.0".
the Y of the a3r6 is "2550421.252689367".

[create a34r6]
a34r6 is a area. "An area (34) in r6".
the printed name of the a34r6 is "Room 6".
Understand "Area 34 in Room 6" as a34r6.
the parent of the a34r6 is r6.
the X of the a34r6 is "547436.8207015463".
the Y of the a34r6 is "2551510.4533167714".

[create a35r6]
a35r6 is a area. "An area (35) in r6".
the printed name of the a35r6 is "Room 6".
Understand "Area 35 in Room 6" as a35r6.
the parent of the a35r6 is r6.
the X of the a35r6 is "546766.0".
the Y of the a35r6 is "2550400.0".

[create a36r6]
a36r6 is a area. "An area (36) in r6".
the printed name of the a36r6 is "Room 6".
Understand "Area 36 in Room 6" as a36r6.
the parent of the a36r6 is r6.
the X of the a36r6 is "546731.0999999999".
the Y of the a36r6 is "2550473.628215993".

[create a37r6]
a37r6 is a area. "An area (37) in r6".
the printed name of the a37r6 is "Room 6".
Understand "Area 37 in Room 6" as a37r6.
the parent of the a37r6 is r6.
the X of the a37r6 is "548073.0".
the Y of the a37r6 is "2550400.0".

[create a38r6]
a38r6 is a area. "An area (38) in r6".
the printed name of the a38r6 is "Room 6".
Understand "Area 38 in Room 6" as a38r6.
the parent of the a38r6 is r6.
the X of the a38r6 is "548109.0".
the Y of the a38r6 is "2550470.636207741".

[create a54r6]
a54r6 is a area. "An area (54) in r6".
the printed name of the a54r6 is "Room 6".
Understand "Area 54 in Room 6" as a54r6.
the parent of the a54r6 is r6.
the X of the a54r6 is "546417.0".
the Y of the a54r6 is "2550400.0".

[create a55r6]
a55r6 is a area. "An area (55) in r6".
the printed name of the a55r6 is "Room 6".
Understand "Area 55 in Room 6" as a55r6.
the parent of the a55r6 is r6.
the X of the a55r6 is "548433.0".
the Y of the a55r6 is "2550400.0".

[create d0]
d0 is a door. "A door between a29r1 and a34r6".
d0 is south of a29r1 and north of a34r6.

[create d2]
d2 is a door. "A door between a19r3 and a31r4".
d2 is south of a19r3 and north of a31r4.

[create d4]
d4 is a door. "A door between a7r4 and a11r5".
d4 is north of a7r4 and south of a11r5.

[create d6]
d6 is a door. "A door between a24r1 and a25r2".
d6 is south of a24r1 and north of a25r2.

[create d8]
d8 is a door. "A door between a26r2 and a22r3".
d8 is north of a26r2 and south of a22r3.

[create d10]
d10 is a door. "A door between a27r0 and a28r1".
d10 is northeast of a27r0 and southwest of a28r1.

[create d11]
d11 is a door. "A door between a27r0 and a14r1".
d11 is northwest of a27r0 and southeast of a14r1.

northeast of a0r6 is southwest of a1r6.

southeast of a2r6 is northwest of a3r6.

northeast of a4r0 is southwest of a5r0.

southeast of a5r0 is northwest of a6r0.

southeast of a7r4 is northwest of a8r4.

southeast of a9r1 is northwest of a10r1.

northeast of a11r5 is southwest of a12r5.

southeast of a13r1 is northwest of a14r1.

southeast of a15r1 is northwest of a16r1.

northeast of a16r1 is southwest of a17r1.

northeast of a10r1 is southwest of a18r1.

northeast of a19r3 is southwest of a20r3.

southeast of a21r3 is northwest of a22r3.

west of a19r3 is east of a22r3.

northwest of a11r5 is southeast of a23r5.

southeast of a28r1 is northwest of a29r1.

southwest of a24r1 is northeast of a29r1.

south of a10r1 is north of a24r1.

northeast of a30r4 is southwest of a31r4.

southwest of a25r2 is northeast of a32r2.

southeast of a26r2 is northwest of a33r2.

northeast of a1r6 is southwest of a34r6.

northwest of a2r6 is southeast of a34r6.

north of a5r0 is south of a27r0.

northwest of a35r6 is southeast of a36r6.

northeast of a37r6 is southwest of a38r6.

southwest of a39r1 is northeast of a40r1.

northeast of a20r3 is southwest of a41r3.

northwest of a21r3 is southeast of a42r3.

northwest of a23r5 is southeast of a43r5.

northwest of a9r1 is southeast of a44r1.

northeast of a12r5 is southwest of a45r5.

southeast of a8r4 is northwest of a46r4.

southwest of a30r4 is northeast of a47r4.

southwest of a32r2 is northeast of a48r2.

southeast of a33r2 is northwest of a49r2.

northwest of a13r1 is southeast of a50r1.

northwest of a15r1 is southeast of a51r1.

northeast of a17r1 is southwest of a52r1.

northeast of a18r1 is southwest of a53r1.

southwest of a0r6 is northeast of a54r6.

southeast of a3r6 is northwest of a55r6.

southwest of a4r0 is northeast of a56r0.

southeast of a6r0 is northwest of a57r0.

west of a7r4 is east of a31r4.

east of a25r2 is west of a26r2.

south of a16r1 is north of a28r1.

northeast of a14r1 is southwest of a40r1.

southeast of a1r6 is northwest of a36r6.

southwest of a2r6 is northeast of a38r6.

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
the player is in a45r5.

the orientation of the player is 6.
