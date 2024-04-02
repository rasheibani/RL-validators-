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

[create a2r0]
a2r0 is a area. "An area (2) in r0".
the printed name of the a2r0 is "Room 0".
Understand "Area 2 in Room 0" as a2r0.
the parent of the a2r0 is r0.
the X of the a2r0 is "545386.2064445317".
the Y of the a2r0 is "2541095.670986919".

[create a3r0]
a3r0 is a area. "An area (3) in r0".
the printed name of the a3r0 is "Room 0".
Understand "Area 3 in Room 0" as a3r0.
the parent of the a3r0 is r0.
the X of the a3r0 is "546027.6855935078".
the Y of the a3r0 is "2540230.6292565824".

[create a4r0]
a4r0 is a area. "An area (4) in r0".
the printed name of the a4r0 is "Room 0".
Understand "Area 4 in Room 0" as a4r0.
the parent of the a4r0 is r0.
the X of the a4r0 is "546903.0535560314".
the Y of the a4r0 is "2541097.1438332587".

[create a5r0]
a5r0 is a area. "An area (5) in r0".
the printed name of the a5r0 is "Room 0".
Understand "Area 5 in Room 0" as a5r0.
the parent of the a5r0 is r0.
the X of the a5r0 is "547543.6855935078".
the Y of the a5r0 is "2540230.6292565824".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "546714.5752428058".
the Y of the a6r0 is "2541107.229480308".

[create a7r0]
a7r0 is a area. "An area (7) in r0".
the printed name of the a7r0 is "Room 0".
Understand "Area 7 in Room 0" as a7r0.
the parent of the a7r0 is r0.
the X of the a7r0 is "548226.4604242068".
the Y of the a7r0 is "2541005.0055419225".

[create a8r0]
a8r0 is a area. "An area (8) in r0".
the printed name of the a8r0 is "Room 0".
Understand "Area 8 in Room 0" as a8r0.
the parent of the a8r0 is r0.
the X of the a8r0 is "548241.5078295618".
the Y of the a8r0 is "2541081.6406066758".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "545148.6999999995".
the Y of the a9r0 is "2541217.277040732".

[create a10r0]
a10r0 is a area. "An area (10) in r0".
the printed name of the a10r0 is "Room 0".
Understand "Area 10 in Room 0" as a10r0.
the parent of the a10r0 is r0.
the X of the a10r0 is "546641.4999999998".
the Y of the a10r0 is "2541236.290050312".

[create a30r0]
a30r0 is a area. "An area (30) in r0".
the printed name of the a30r0 is "Room 0".
Understand "Area 30 in Room 0" as a30r0.
the parent of the a30r0 is r0.
the X of the a30r0 is "546973.5000000031".
the Y of the a30r0 is "2541190.5088652493".

[create a31r0]
a31r0 is a area. "An area (31) in r0".
the printed name of the a31r0 is "Room 0".
Understand "Area 31 in Room 0" as a31r0.
the parent of the a31r0 is r0.
the X of the a31r0 is "548456.5".
the Y of the a31r0 is "2541214.3790818145".

[create a35r0]
a35r0 is a area. "An area (35) in r0".
the printed name of the a35r0 is "Room 0".
Understand "Area 35 in Room 0" as a35r0.
the parent of the a35r0 is r0.
the X of the a35r0 is "548094.75".
the Y of the a35r0 is "2541220.492935174".

[create a41r0]
a41r0 is a area. "An area (41) in r0".
the printed name of the a41r0 is "Room 0".
Understand "Area 41 in Room 0" as a41r0.
the parent of the a41r0 is r0.
the X of the a41r0 is "545461.0".
the Y of the a41r0 is "2541240.0".

[create a42r0]
a42r0 is a area. "An area (42) in r0".
the printed name of the a42r0 is "Room 0".
Understand "Area 42 in Room 0" as a42r0.
the parent of the a42r0 is r0.
the X of the a42r0 is "545426.2999999997".
the Y of the a42r0 is "2541168.436524822".

[create a45r0]
a45r0 is a area. "An area (45) in r0".
the printed name of the a45r0 is "Room 0".
Understand "Area 45 in Room 0" as a45r0.
the parent of the a45r0 is r0.
the X of the a45r0 is "548070.0".
the Y of the a45r0 is "2541240.0".

[create a47r0]
a47r0 is a area. "An area (47) in r0".
the printed name of the a47r0 is "Room 0".
Understand "Area 47 in Room 0" as a47r0.
the parent of the a47r0 is r0.
the X of the a47r0 is "546977.0".
the Y of the a47r0 is "2541240.0".

[create a52r0]
a52r0 is a area. "An area (52) in r0".
the printed name of the a52r0 is "Room 0".
Understand "Area 52 in Room 0" as a52r0.
the parent of the a52r0 is r0.
the X of the a52r0 is "545114.0".
the Y of the a52r0 is "2541240.0".

[create a53r0]
a53r0 is a area. "An area (53) in r0".
the printed name of the a53r0 is "Room 0".
Understand "Area 53 in Room 0" as a53r0.
the parent of the a53r0 is r0.
the X of the a53r0 is "546630.0".
the Y of the a53r0 is "2541240.0".

[create a54r0]
a54r0 is a area. "An area (54) in r0".
the printed name of the a54r0 is "Room 0".
Understand "Area 54 in Room 0" as a54r0.
the parent of the a54r0 is r0.
the X of the a54r0 is "548486.0".
the Y of the a54r0 is "2541240.0".

[create a18r1]
a18r1 is a area. "An area (18) in r1".
the printed name of the a18r1 is "Room 1".
Understand "Area 18 in Room 1" as a18r1.
the parent of the a18r1 is r1.
the X of the a18r1 is "548770.1566116108".
the Y of the a18r1 is "2539153.489403779".

[create a19r1]
a19r1 is a area. "An area (19) in r1".
the printed name of the a19r1 is "Room 1".
Understand "Area 19 in Room 1" as a19r1.
the parent of the a19r1 is r1.
the X of the a19r1 is "548832.9412801259".
the Y of the a19r1 is "2539154.6105585736".

[create a28r1]
a28r1 is a area. "An area (28) in r1".
the printed name of the a28r1 is "Room 1".
Understand "Area 28 in Room 1" as a28r1.
the parent of the a28r1 is r1.
the X of the a28r1 is "548690.125".
the Y of the a28r1 is "2539050.1645021765".

[create a29r1]
a29r1 is a area. "An area (29) in r1".
the printed name of the a29r1 is "Room 1".
Understand "Area 29 in Room 1" as a29r1.
the parent of the a29r1 is r1.
the X of the a29r1 is "548912.875".
the Y of the a29r1 is "2539050.099474627".

[create a50r1]
a50r1 is a area. "An area (50) in r1".
the printed name of the a50r1 is "Room 1".
Understand "Area 50 in Room 1" as a50r1.
the parent of the a50r1 is r1.
the X of the a50r1 is "548653.0".
the Y of the a50r1 is "2539000.0".

[create a51r1]
a51r1 is a area. "An area (51) in r1".
the printed name of the a51r1 is "Room 1".
Understand "Area 51 in Room 1" as a51r1.
the parent of the a51r1 is r1.
the X of the a51r1 is "548950.0".
the Y of the a51r1 is "2539000.0".

[create a14r2]
a14r2 is a area. "An area (14) in r2".
the printed name of the a14r2 is "Room 2".
Understand "Area 14 in Room 2" as a14r2.
the parent of the a14r2 is r2.
the X of the a14r2 is "549555.8305982081".
the Y of the a14r2 is "2541098.8947038352".

[create a20r2]
a20r2 is a area. "An area (20) in r2".
the printed name of the a20r2 is "Room 2".
Understand "Area 20 in Room 2" as a20r2.
the parent of the a20r2 is r2.
the X of the a20r2 is "549419.5565103933".
the Y of the a20r2 is "2541102.537885611".

[create a32r2]
a32r2 is a area. "An area (32) in r2".
the printed name of the a32r2 is "Room 2".
Understand "Area 32 in Room 2" as a32r2.
the parent of the a32r2 is r2.
the X of the a32r2 is "549628.8999999999".
the Y of the a32r2 is "2541189.466754617".

[create a33r2]
a33r2 is a area. "An area (33) in r2".
the printed name of the a33r2 is "Room 2".
Understand "Area 33 in Room 2" as a33r2.
the parent of the a33r2 is r2.
the X of the a33r2 is "549348.1000000006".
the Y of the a33r2 is "2541189.523978246".

[create a43r2]
a43r2 is a area. "An area (43) in r2".
the printed name of the a43r2 is "Room 2".
Understand "Area 43 in Room 2" as a43r2.
the parent of the a43r2 is r2.
the X of the a43r2 is "549313.0".
the Y of the a43r2 is "2541240.0".

[create a55r2]
a55r2 is a area. "An area (55) in r2".
the printed name of the a55r2 is "Room 2".
Understand "Area 55 in Room 2" as a55r2.
the parent of the a55r2 is r2.
the X of the a55r2 is "549664.0".
the Y of the a55r2 is "2541240.0".

[create a15r3]
a15r3 is a area. "An area (15) in r3".
the printed name of the a15r3 is "Room 3".
Understand "Area 15 in Room 3" as a15r3.
the parent of the a15r3 is r3.
the X of the a15r3 is "550129.2576810804".
the Y of the a15r3 is "2539153.704645209".

[create a16r3]
a16r3 is a area. "An area (16) in r3".
the printed name of the a16r3 is "Room 3".
Understand "Area 16 in Room 3" as a16r3.
the parent of the a16r3 is r3.
the X of the a16r3 is "550178.0751328206".
the Y of the a16r3 is "2539154.3386380887".

[create a26r3]
a26r3 is a area. "An area (26) in r3".
the printed name of the a26r3 is "Room 3".
Understand "Area 26 in Room 3" as a26r3.
the parent of the a26r3 is r3.
the X of the a26r3 is "550259.0".
the Y of the a26r3 is "2539049.4606324406".

[create a27r3]
a27r3 is a area. "An area (27) in r3".
the printed name of the a27r3 is "Room 3".
Understand "Area 27 in Room 3" as a27r3.
the parent of the a27r3 is r3.
the X of the a27r3 is "550049.0".
the Y of the a27r3 is "2539049.4226234336".

[create a48r3]
a48r3 is a area. "An area (48) in r3".
the printed name of the a48r3 is "Room 3".
Understand "Area 48 in Room 3" as a48r3.
the parent of the a48r3 is r3.
the X of the a48r3 is "550294.0".
the Y of the a48r3 is "2539000.0".

[create a49r3]
a49r3 is a area. "An area (49) in r3".
the printed name of the a49r3 is "Room 3".
Understand "Area 49 in Room 3" as a49r3.
the parent of the a49r3 is r3.
the X of the a49r3 is "550014.0".
the Y of the a49r3 is "2539000.0".

[create a17r4]
a17r4 is a area. "An area (17) in r4".
the printed name of the a17r4 is "Room 4".
Understand "Area 17 in Room 4" as a17r4.
the parent of the a17r4 is r4.
the X of the a17r4 is "550715.0274643081".
the Y of the a17r4 is "2541102.3687288053".

[create a34r4]
a34r4 is a area. "An area (34) in r4".
the printed name of the a34r4 is "Room 4".
Understand "Area 34 in Room 4" as a34r4.
the parent of the a34r4 is r4.
the X of the a34r4 is "550641.5000000005".
the Y of the a34r4 is "2541191.12906226".

[create a36r4]
a36r4 is a area. "An area (36) in r4".
the printed name of the a36r4 is "Room 4".
Understand "Area 36 in Room 4" as a36r4.
the parent of the a36r4 is r4.
the X of the a36r4 is "550860.4999999999".
the Y of the a36r4 is "2541209.786912203".

[create a44r4]
a44r4 is a area. "An area (44) in r4".
the printed name of the a44r4 is "Room 4".
Understand "Area 44 in Room 4" as a44r4.
the parent of the a44r4 is r4.
the X of the a44r4 is "550605.0".
the Y of the a44r4 is "2541240.0".

[create a46r4]
a46r4 is a area. "An area (46) in r4".
the printed name of the a46r4 is "Room 4".
Understand "Area 46 in Room 4" as a46r4.
the parent of the a46r4 is r4.
the X of the a46r4 is "550897.0".
the Y of the a46r4 is "2541240.0".

[create a0r5]
a0r5 is a area. "An area (0) in r5".
the printed name of the a0r5 is "Room 5".
Understand "Area 0 in Room 5" as a0r5.
the parent of the a0r5 is r5.
the X of the a0r5 is "546033.8207015462".
the Y of the a0r5 is "2540110.4533167714".

[create a1r5]
a1r5 is a area. "An area (1) in r5".
the printed name of the a1r5 is "Room 5".
Understand "Area 1 in Room 5" as a1r5.
the parent of the a1r5 is r5.
the X of the a1r5 is "546759.2508809658".
the Y of the a1r5 is "2539174.9435156053".

[create a11r5]
a11r5 is a area. "An area (11) in r5".
the printed name of the a11r5 is "Room 5".
Understand "Area 11 in Room 5" as a11r5.
the parent of the a11r5 is r5.
the X of the a11r5 is "545290.2994279689".
the Y of the a11r5 is "2539142.112798646".

[create a12r5]
a12r5 is a area. "An area (12) in r5".
the printed name of the a12r5 is "Room 5".
Understand "Area 12 in Room 5" as a12r5.
the parent of the a12r5 is r5.
the X of the a12r5 is "547549.8207015463".
the Y of the a12r5 is "2540110.4533167714".

[create a13r5]
a13r5 is a area. "An area (13) in r5".
the printed name of the a13r5 is "Room 5".
Understand "Area 13 in Room 5" as a13r5.
the parent of the a13r5 is r5.
the X of the a13r5 is "548262.7281516471".
the Y of the a13r5 is "2539147.941871377".

[create a21r5]
a21r5 is a area. "An area (21) in r5".
the printed name of the a21r5 is "Room 5".
Understand "Area 21 in Room 5" as a21r5.
the parent of the a21r5 is r5.
the X of the a21r5 is "546794.1726802315".
the Y of the a21r5 is "2539174.401430309".

[create a22r5]
a22r5 is a area. "An area (22) in r5".
the printed name of the a22r5 is "Room 5".
Understand "Area 22 in Room 5" as a22r5.
the parent of the a22r5 is r5.
the X of the a22r5 is "546565.0".
the Y of the a22r5 is "2539011.860408382".

[create a23r5]
a23r5 is a area. "An area (23) in r5".
the printed name of the a23r5 is "Room 5".
Understand "Area 23 in Room 5" as a23r5.
the parent of the a23r5 is r5.
the X of the a23r5 is "546992.25".
the Y of the a23r5 is "2539013.094044855".

[create a24r5]
a24r5 is a area. "An area (24) in r5".
the printed name of the a24r5 is "Room 5".
Understand "Area 24 in Room 5" as a24r5.
the parent of the a24r5 is r5.
the X of the a24r5 is "545048.8999999999".
the Y of the a24r5 is "2539021.5829838095".

[create a25r5]
a25r5 is a area. "An area (25) in r5".
the printed name of the a25r5 is "Room 5".
Understand "Area 25 in Room 5" as a25r5.
the parent of the a25r5 is r5.
the X of the a25r5 is "548510.0".
the Y of the a25r5 is "2539021.2526893676".

[create a37r5]
a37r5 is a area. "An area (37) in r5".
the printed name of the a37r5 is "Room 5".
Understand "Area 37 in Room 5" as a37r5.
the parent of the a37r5 is r5.
the X of the a37r5 is "545363.0".
the Y of the a37r5 is "2539000.0".

[create a38r5]
a38r5 is a area. "An area (38) in r5".
the printed name of the a38r5 is "Room 5".
Understand "Area 38 in Room 5" as a38r5.
the parent of the a38r5 is r5.
the X of the a38r5 is "545328.0999999999".
the Y of the a38r5 is "2539073.628215993".

[create a39r5]
a39r5 is a area. "An area (39) in r5".
the printed name of the a39r5 is "Room 5".
Understand "Area 39 in Room 5" as a39r5.
the parent of the a39r5 is r5.
the X of the a39r5 is "548186.0".
the Y of the a39r5 is "2539000.0".

[create a40r5]
a40r5 is a area. "An area (40) in r5".
the printed name of the a40r5 is "Room 5".
Understand "Area 40 in Room 5" as a40r5.
the parent of the a40r5 is r5.
the X of the a40r5 is "548222.0".
the Y of the a40r5 is "2539070.636207741".

[create a56r5]
a56r5 is a area. "An area (56) in r5".
the printed name of the a56r5 is "Room 5".
Understand "Area 56 in Room 5" as a56r5.
the parent of the a56r5 is r5.
the X of the a56r5 is "546530.0".
the Y of the a56r5 is "2539000.0".

[create a57r5]
a57r5 is a area. "An area (57) in r5".
the printed name of the a57r5 is "Room 5".
Understand "Area 57 in Room 5" as a57r5.
the parent of the a57r5 is r5.
the X of the a57r5 is "547030.0".
the Y of the a57r5 is "2539000.0".

[create a58r5]
a58r5 is a area. "An area (58) in r5".
the printed name of the a58r5 is "Room 5".
Understand "Area 58 in Room 5" as a58r5.
the parent of the a58r5 is r5.
the X of the a58r5 is "545014.0".
the Y of the a58r5 is "2539000.0".

[create a59r5]
a59r5 is a area. "An area (59) in r5".
the printed name of the a59r5 is "Room 5".
Understand "Area 59 in Room 5" as a59r5.
the parent of the a59r5 is r5.
the X of the a59r5 is "548546.0".
the Y of the a59r5 is "2539000.0".

[create d0]
d0 is a door. "A door between a3r0 and a0r5".
d0 is south of a3r0 and north of a0r5.

[create d1]
d1 is a door. "A door between a5r0 and a12r5".
d1 is south of a5r0 and north of a12r5.

[create d4]
d4 is a door. "A door between a14r2 and a15r3".
d4 is south of a14r2 and north of a15r3.

[create d6]
d6 is a door. "A door between a16r3 and a17r4".
d6 is north of a16r3 and south of a17r4.

[create d8]
d8 is a door. "A door between a7r0 and a18r1".
d8 is south of a7r0 and north of a18r1.

[create d10]
d10 is a door. "A door between a19r1 and a20r2".
d10 is north of a19r1 and south of a20r2.

southeast of a0r5 is northwest of a1r5.

southeast of a2r0 is northwest of a3r0.

southeast of a4r0 is northwest of a5r0.

northeast of a3r0 is southwest of a6r0.

northeast of a5r0 is southwest of a7r0.

north of a7r0 is south of a8r0.

northwest of a2r0 is southeast of a9r0.

northwest of a6r0 is southeast of a10r0.

west of a4r0 is east of a6r0.

southwest of a0r5 is northeast of a11r5.

southeast of a12r5 is northwest of a13r5.

east of a1r5 is west of a21r5.

southwest of a12r5 is northeast of a21r5.

southwest of a1r5 is northeast of a22r5.

southeast of a21r5 is northwest of a23r5.

southwest of a11r5 is northeast of a24r5.

southeast of a13r5 is northwest of a25r5.

southeast of a16r3 is northwest of a26r3.

southwest of a15r3 is northeast of a27r3.

southwest of a18r1 is northeast of a28r1.

southeast of a19r1 is northwest of a29r1.

northeast of a4r0 is southwest of a30r0.

northeast of a8r0 is southwest of a31r0.

northeast of a14r2 is southwest of a32r2.

northwest of a20r2 is southeast of a33r2.

west of a14r2 is east of a20r2.

northwest of a17r4 is southeast of a34r4.

northwest of a8r0 is southeast of a35r0.

northeast of a17r4 is southwest of a36r4.

northwest of a37r5 is southeast of a38r5.

northeast of a39r5 is southwest of a40r5.

southwest of a41r0 is northeast of a42r0.

northwest of a33r2 is southeast of a43r2.

northwest of a34r4 is southeast of a44r4.

northwest of a35r0 is southeast of a45r0.

northeast of a36r4 is southwest of a46r4.

north of a30r0 is south of a47r0.

southeast of a26r3 is northwest of a48r3.

southwest of a27r3 is northeast of a49r3.

southwest of a28r1 is northeast of a50r1.

southeast of a29r1 is northwest of a51r1.

northwest of a9r0 is southeast of a52r0.

west of a10r0 is east of a53r0.

northeast of a31r0 is southwest of a54r0.

northeast of a32r2 is southwest of a55r2.

west of a22r5 is east of a56r5.

east of a23r5 is west of a57r5.

southwest of a24r5 is northeast of a58r5.

southeast of a25r5 is northwest of a59r5.

southwest of a13r5 is northeast of a40r5.

east of a15r3 is west of a16r3.

east of a18r1 is west of a19r1.

northeast of a2r0 is southwest of a42r0.

southeast of a11r5 is northwest of a38r5.

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
the player is in a10r0.

the orientation of the player is 6.
