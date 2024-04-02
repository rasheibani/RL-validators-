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

[create r7]
r7 is a indoor_room. "Room 7".

[create r8]
r8 is a indoor_room. "Room 8".

[create a5r0]
a5r0 is a area. "An area (5) in r0".
the printed name of the a5r0 is "Room 0".
Understand "Area 5 in Room 0" as a5r0.
the parent of the a5r0 is r0.
the X of the a5r0 is "545271.375".
the Y of the a5r0 is "2225801.379310345".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "545427.2999389705".
the Y of the a6r0 is "2225645.581246234".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "545449.7851686038".
the Y of the a23r0 is "2225569.863389253".

[create a40r0]
a40r0 is a area. "An area (40) in r0".
the printed name of the a40r0 is "Room 0".
Understand "Area 40 in Room 0" as a40r0.
the parent of the a40r0 is r0.
the X of the a40r0 is "545533.0".
the Y of the a40r0 is "2225840.0".

[create a41r0]
a41r0 is a area. "An area (41) in r0".
the printed name of the a41r0 is "Room 0".
Understand "Area 41 in Room 0" as a41r0.
the parent of the a41r0 is r0.
the X of the a41r0 is "545495.6249999999".
the Y of the a41r0 is "2225768.8402001346".

[create a47r0]
a47r0 is a area. "An area (47) in r0".
the printed name of the a47r0 is "Room 0".
Understand "Area 47 in Room 0" as a47r0.
the parent of the a47r0 is r0.
the X of the a47r0 is "545234.0".
the Y of the a47r0 is "2225840.0".

[create a16r1]
a16r1 is a area. "An area (16) in r1".
the printed name of the a16r1 is "Room 1".
Understand "Area 16 in Room 1" as a16r1.
the parent of the a16r1 is r1.
the X of the a16r1 is "546749.9107333675".
the Y of the a16r1 is "2223871.1586977267".

[create a29r1]
a29r1 is a area. "An area (29) in r1".
the printed name of the a29r1 is "Room 1".
Understand "Area 29 in Room 1" as a29r1.
the parent of the a29r1 is r1.
the X of the a29r1 is "546773.3030177355".
the Y of the a29r1 is "2223793.6294527887".

[create a32r1]
a32r1 is a area. "An area (32) in r1".
the printed name of the a32r1 is "Room 1".
Understand "Area 32 in Room 1" as a32r1.
the parent of the a32r1 is r1.
the X of the a32r1 is "546929.625".
the Y of the a32r1 is "2223637.2918978916".

[create a38r1]
a38r1 is a area. "An area (38) in r1".
the printed name of the a38r1 is "Room 1".
Understand "Area 38 in Room 1" as a38r1.
the parent of the a38r1 is r1.
the X of the a38r1 is "546668.0".
the Y of the a38r1 is "2223600.0".

[create a39r1]
a39r1 is a area. "An area (39) in r1".
the printed name of the a39r1 is "Room 1".
Understand "Area 39 in Room 1" as a39r1.
the parent of the a39r1 is r1.
the X of the a39r1 is "546705.375".
the Y of the a39r1 is "2223671.1563946763".

[create a55r1]
a55r1 is a area. "An area (55) in r1".
the printed name of the a55r1 is "Room 1".
Understand "Area 55 in Room 1" as a55r1.
the parent of the a55r1 is r1.
the X of the a55r1 is "546967.0".
the Y of the a55r1 is "2223600.0".

[create a8r2]
a8r2 is a area. "An area (8) in r2".
the printed name of the a8r2 is "Room 2".
Understand "Area 8 in Room 2" as a8r2.
the parent of the a8r2 is r2.
the X of the a8r2 is "546867.2179386405".
the Y of the a8r2 is "2225660.861123786".

[create a9r2]
a9r2 is a area. "An area (9) in r2".
the printed name of the a9r2 is "Room 2".
Understand "Area 9 in Room 2" as a9r2.
the parent of the a9r2 is r2.
the X of the a9r2 is "546989.5".
the Y of the a9r2 is "2225796.173420517".

[create a11r2]
a11r2 is a area. "An area (11) in r2".
the printed name of the a11r2 is "Room 2".
Understand "Area 11 in Room 2" as a11r2.
the parent of the a11r2 is r2.
the X of the a11r2 is "546701.0000000005".
the Y of the a11r2 is "2225801.7608695654".

[create a17r2]
a17r2 is a area. "An area (17) in r2".
the printed name of the a17r2 is "Room 2".
Understand "Area 17 in Room 2" as a17r2.
the parent of the a17r2 is r2.
the X of the a17r2 is "546929.236919429".
the Y of the a17r2 is "2225108.500905724".

[create a49r2]
a49r2 is a area. "An area (49) in r2".
the printed name of the a49r2 is "Room 2".
Understand "Area 49 in Room 2" as a49r2.
the parent of the a49r2 is r2.
the X of the a49r2 is "546967.0".
the Y of the a49r2 is "2225840.0".

[create a50r2]
a50r2 is a area. "An area (50) in r2".
the printed name of the a50r2 is "Room 2".
Understand "Area 50 in Room 2" as a50r2.
the parent of the a50r2 is r2.
the X of the a50r2 is "546714.0".
the Y of the a50r2 is "2225840.0".

[create a25r3]
a25r3 is a area. "An area (25) in r3".
the printed name of the a25r3 is "Room 3".
Understand "Area 25 in Room 3" as a25r3.
the parent of the a25r3 is r3.
the X of the a25r3 is "547414.6789846191".
the Y of the a25r3 is "2223753.498731868".

[create a28r3]
a28r3 is a area. "An area (28) in r3".
the printed name of the a28r3 is "Room 3".
Understand "Area 28 in Room 3" as a28r3.
the parent of the a28r3 is r3.
the X of the a28r3 is "547476.9412801262".
the Y of the a28r3 is "2223754.6105585736".

[create a35r3]
a35r3 is a area. "An area (35) in r3".
the printed name of the a35r3 is "Room 3".
Understand "Area 35 in Room 3" as a35r3.
the parent of the a35r3 is r3.
the X of the a35r3 is "547334.125".
the Y of the a35r3 is "2223649.4804712893".

[create a36r3]
a36r3 is a area. "An area (36) in r3".
the printed name of the a36r3 is "Room 3".
Understand "Area 36 in Room 3" as a36r3.
the parent of the a36r3 is r3.
the X of the a36r3 is "547556.875".
the Y of the a36r3 is "2223650.099474627".

[create a58r3]
a58r3 is a area. "An area (58) in r3".
the printed name of the a58r3 is "Room 3".
Understand "Area 58 in Room 3" as a58r3.
the parent of the a58r3 is r3.
the X of the a58r3 is "547297.0".
the Y of the a58r3 is "2223600.0".

[create a59r3]
a59r3 is a area. "An area (59) in r3".
the printed name of the a59r3 is "Room 3".
Understand "Area 59 in Room 3" as a59r3.
the parent of the a59r3 is r3.
the X of the a59r3 is "547594.0".
the Y of the a59r3 is "2223600.0".

[create a12r4]
a12r4 is a area. "An area (12) in r4".
the printed name of the a12r4 is "Room 4".
Understand "Area 12 in Room 4" as a12r4.
the parent of the a12r4 is r4.
the X of the a12r4 is "548199.8305982079".
the Y of the a12r4 is "2225698.8947038352".

[create a13r4]
a13r4 is a area. "An area (13) in r4".
the printed name of the a13r4 is "Room 4".
Understand "Area 13 in Room 4" as a13r4.
the parent of the a13r4 is r4.
the X of the a13r4 is "548272.9000000001".
the Y of the a13r4 is "2225789.466754617".

[create a14r4]
a14r4 is a area. "An area (14) in r4".
the printed name of the a14r4 is "Room 4".
Understand "Area 14 in Room 4" as a14r4.
the parent of the a14r4 is r4.
the X of the a14r4 is "547992.0999999996".
the Y of the a14r4 is "2225789.5239782464".

[create a15r4]
a15r4 is a area. "An area (15) in r4".
the printed name of the a15r4 is "Room 4".
Understand "Area 15 in Room 4" as a15r4.
the parent of the a15r4 is r4.
the X of the a15r4 is "548063.5565103926".
the Y of the a15r4 is "2225702.5378856105".

[create a42r4]
a42r4 is a area. "An area (42) in r4".
the printed name of the a42r4 is "Room 4".
Understand "Area 42 in Room 4" as a42r4.
the parent of the a42r4 is r4.
the X of the a42r4 is "547957.0".
the Y of the a42r4 is "2225840.0".

[create a51r4]
a51r4 is a area. "An area (51) in r4".
the printed name of the a51r4 is "Room 4".
Understand "Area 51 in Room 4" as a51r4.
the parent of the a51r4 is r4.
the X of the a51r4 is "548308.0".
the Y of the a51r4 is "2225840.0".

[create a24r5]
a24r5 is a area. "An area (24) in r5".
the printed name of the a24r5 is "Room 5".
Understand "Area 24 in Room 5" as a24r5.
the parent of the a24r5 is r5.
the X of the a24r5 is "548773.2576810801".
the Y of the a24r5 is "2223753.7046452085".

[create a26r5]
a26r5 is a area. "An area (26) in r5".
the printed name of the a26r5 is "Room 5".
Understand "Area 26 in Room 5" as a26r5.
the parent of the a26r5 is r5.
the X of the a26r5 is "548821.8614477161".
the Y of the a26r5 is "2223754.335862957".

[create a33r5]
a33r5 is a area. "An area (33) in r5".
the printed name of the a33r5 is "Room 5".
Understand "Area 33 in Room 5" as a33r5.
the parent of the a33r5 is r5.
the X of the a33r5 is "548903.0".
the Y of the a33r5 is "2223649.1748269037".

[create a34r5]
a34r5 is a area. "An area (34) in r5".
the printed name of the a34r5 is "Room 5".
Understand "Area 34 in Room 5" as a34r5.
the parent of the a34r5 is r5.
the X of the a34r5 is "548693.0".
the Y of the a34r5 is "2223649.4226234336".

[create a56r5]
a56r5 is a area. "An area (56) in r5".
the printed name of the a56r5 is "Room 5".
Understand "Area 56 in Room 5" as a56r5.
the parent of the a56r5 is r5.
the X of the a56r5 is "548938.0".
the Y of the a56r5 is "2223600.0".

[create a57r5]
a57r5 is a area. "An area (57) in r5".
the printed name of the a57r5 is "Room 5".
Understand "Area 57 in Room 5" as a57r5.
the parent of the a57r5 is r5.
the X of the a57r5 is "548658.0".
the Y of the a57r5 is "2223600.0".

[create a3r6]
a3r6 is a area. "An area (3) in r6".
the printed name of the a3r6 is "Room 6".
Understand "Area 3 in Room 6" as a3r6.
the parent of the a3r6 is r6.
the X of the a3r6 is "549438.6823243613".
the Y of the a3r6 is "2225552.8492366956".

[create a4r6]
a4r6 is a area. "An area (4) in r6".
the printed name of the a4r6 is "Room 6".
Understand "Area 4 in Room 6" as a4r6.
the parent of the a4r6 is r6.
the X of the a4r6 is "549441.2624832876".
the Y of the a4r6 is "2225594.597593125".

[create a7r6]
a7r6 is a area. "An area (7) in r6".
the printed name of the a7r6 is "Room 6".
Understand "Area 7 in Room 6" as a7r6.
the parent of the a7r6 is r6.
the X of the a7r6 is "549286.0".
the Y of the a7r6 is "2225791.0095196413".

[create a10r6]
a10r6 is a area. "An area (10) in r6".
the printed name of the a10r6 is "Room 6".
Understand "Area 10 in Room 6" as a10r6.
the parent of the a10r6 is r6.
the X of the a10r6 is "550750.8400677575".
the Y of the a10r6 is "2225712.773180337".

[create a18r6]
a18r6 is a area. "An area (18) in r6".
the printed name of the a18r6 is "Room 6".
Understand "Area 18 in Room 6" as a18r6.
the parent of the a18r6 is r6.
the X of the a18r6 is "550590.7198228128".
the Y of the a18r6 is "2224752.0".

[create a20r6]
a20r6 is a area. "An area (20) in r6".
the printed name of the a20r6 is "Room 6".
Understand "Area 20 in Room 6" as a20r6.
the parent of the a20r6 is r6.
the X of the a20r6 is "549570.4599293644".
the Y of the a20r6 is "2224752.0".

[create a21r6]
a21r6 is a area. "An area (21) in r6".
the printed name of the a21r6 is "Room 6".
Understand "Area 21 in Room 6" as a21r6.
the parent of the a21r6 is r6.
the X of the a21r6 is "550683.142857143".
the Y of the a21r6 is "2224848.0".

[create a22r6]
a22r6 is a area. "An area (22) in r6".
the printed name of the a22r6 is "Room 6".
Understand "Area 22 in Room 6" as a22r6.
the parent of the a22r6 is r6.
the X of the a22r6 is "550843.71875".
the Y of the a22r6 is "2225616.0".

[create a37r6]
a37r6 is a area. "An area (37) in r6".
the printed name of the a37r6 is "Room 6".
Understand "Area 37 in Room 6" as a37r6.
the parent of the a37r6 is r6.
the X of the a37r6 is "550841.5882352942".
the Y of the a37r6 is "2225808.0".

[create a43r6]
a43r6 is a area. "An area (43) in r6".
the printed name of the a43r6 is "Room 6".
Understand "Area 43 in Room 6" as a43r6.
the parent of the a43r6 is r6.
the X of the a43r6 is "550881.0".
the Y of the a43r6 is "2225840.0".

[create a45r6]
a45r6 is a area. "An area (45) in r6".
the printed name of the a45r6 is "Room 6".
Understand "Area 45 in Room 6" as a45r6.
the parent of the a45r6 is r6.
the X of the a45r6 is "550720.0".
the Y of the a45r6 is "2224880.0".

[create a46r6]
a46r6 is a area. "An area (46) in r6".
the printed name of the a46r6 is "Room 6".
Understand "Area 46 in Room 6" as a46r6.
the parent of the a46r6 is r6.
the X of the a46r6 is "550881.0".
the Y of the a46r6 is "2225584.0".

[create a48r6]
a48r6 is a area. "An area (48) in r6".
the printed name of the a48r6 is "Room 6".
Understand "Area 48 in Room 6" as a48r6.
the parent of the a48r6 is r6.
the X of the a48r6 is "549249.0".
the Y of the a48r6 is "2225840.0".

[create a19r7]
a19r7 is a area. "An area (19) in r7".
the printed name of the a19r7 is "Room 7".
Understand "Area 19 in Room 7" as a19r7.
the parent of the a19r7 is r7.
the X of the a19r7 is "550683.1428571428".
the Y of the a19r7 is "2224656.0".

[create a27r7]
a27r7 is a area. "An area (27) in r7".
the printed name of the a27r7 is "Room 7".
Understand "Area 27 in Room 7" as a27r7.
the parent of the a27r7 is r7.
the X of the a27r7 is "549542.5904460568".
the Y of the a27r7 is "2223745.8764311024".

[create a30r7]
a30r7 is a area. "An area (30) in r7".
the printed name of the a30r7 is "Room 7".
Understand "Area 30 in Room 7" as a30r7.
the parent of the a30r7 is r7.
the X of the a30r7 is "549651.625".
the Y of the a30r7 is "2223639.3846153845".

[create a31r7]
a31r7 is a area. "An area (31) in r7".
the printed name of the a31r7 is "Room 7".
Understand "Area 31 in Room 7" as a31r7.
the parent of the a31r7 is r7.
the X of the a31r7 is "549433.375".
the Y of the a31r7 is "2223638.751696065".

[create a44r7]
a44r7 is a area. "An area (44) in r7".
the printed name of the a44r7 is "Room 7".
Understand "Area 44 in Room 7" as a44r7.
the parent of the a44r7 is r7.
the X of the a44r7 is "550720.0".
the Y of the a44r7 is "2224624.0".

[create a53r7]
a53r7 is a area. "An area (53) in r7".
the printed name of the a53r7 is "Room 7".
Understand "Area 53 in Room 7" as a53r7.
the parent of the a53r7 is r7.
the X of the a53r7 is "549688.0".
the Y of the a53r7 is "2223600.0".

[create a54r7]
a54r7 is a area. "An area (54) in r7".
the printed name of the a54r7 is "Room 7".
Understand "Area 54 in Room 7" as a54r7.
the parent of the a54r7 is r7.
the X of the a54r7 is "549397.0".
the Y of the a54r7 is "2223600.0".

[create a0r8]
a0r8 is a area. "An area (0) in r8".
the printed name of the a0r8 is "Room 8".
Understand "Area 0 in Room 8" as a0r8.
the parent of the a0r8 is r8.
the X of the a0r8 is "545268.875".
the Y of the a0r8 is "2223638.6206896557".

[create a1r8]
a1r8 is a area. "An area (1) in r8".
the printed name of the a1r8 is "Room 8".
Understand "Area 1 in Room 8" as a1r8.
the parent of the a1r8 is r8.
the X of the a1r8 is "545373.4298420025".
the Y of the a1r8 is "2223740.163209851".

[create a2r8]
a2r8 is a area. "An area (2) in r8".
the printed name of the a2r8 is "Room 8".
Understand "Area 2 in Room 8" as a2r8.
the parent of the a2r8 is r8.
the X of the a2r8 is "545478.125".
the Y of the a2r8 is "2223638.260869565".

[create a52r8]
a52r8 is a area. "An area (52) in r8".
the printed name of the a52r8 is "Room 8".
Understand "Area 52 in Room 8" as a52r8.
the parent of the a52r8 is r8.
the X of the a52r8 is "545513.0".
the Y of the a52r8 is "2223600.0".

[create a60r8]
a60r8 is a area. "An area (60) in r8".
the printed name of the a60r8 is "Room 8".
Understand "Area 60 in Room 8" as a60r8.
the parent of the a60r8 is r8.
the X of the a60r8 is "545234.0".
the Y of the a60r8 is "2223600.0".

[create d0]
d0 is a door. "A door between a12r4 and a24r5".
d0 is south of a12r4 and north of a24r5.

[create d2]
d2 is a door. "A door between a23r0 and a1r8".
d2 is south of a23r0 and north of a1r8.

[create d4]
d4 is a door. "A door between a17r2 and a25r3".
d4 is south of a17r2 and north of a25r3.

[create d6]
d6 is a door. "A door between a26r5 and a3r6".
d6 is north of a26r5 and south of a3r6.

[create d8]
d8 is a door. "A door between a18r6 and a19r7".
d8 is southeast of a18r6 and northwest of a19r7.

[create d9]
d9 is a door. "A door between a20r6 and a27r7".
d9 is south of a20r6 and north of a27r7.

[create d10]
d10 is a door. "A door between a28r3 and a15r4".
d10 is north of a28r3 and south of a15r4.

[create d12]
d12 is a door. "A door between a23r0 and a16r1".
d12 is southeast of a23r0 and northwest of a16r1.

[create d14]
d14 is a door. "A door between a16r1 and a17r2".
d14 is northeast of a16r1 and southwest of a17r2.

northeast of a0r8 is southwest of a1r8.

southeast of a1r8 is northwest of a2r8.

north of a3r6 is south of a4r6.

southeast of a5r0 is northwest of a6r0.

northwest of a4r6 is southeast of a7r6.

northeast of a8r2 is southwest of a9r2.

east of a4r6 is west of a10r6.

northwest of a8r2 is southeast of a11r2.

northeast of a12r4 is southwest of a13r4.

southeast of a14r4 is northwest of a15r4.

west of a12r4 is east of a15r4.

west of a18r6 is east of a20r6.

northeast of a18r6 is southwest of a21r6.

south of a3r6 is north of a20r6.

south of a8r2 is north of a17r2.

southeast of a10r6 is northwest of a22r6.

south of a6r0 is north of a23r0.

south of a16r1 is north of a29r1.

southeast of a27r7 is northwest of a30r7.

southwest of a27r7 is northeast of a31r7.

southeast of a29r1 is northwest of a32r1.

southeast of a26r5 is northwest of a33r5.

southwest of a24r5 is northeast of a34r5.

southwest of a25r3 is northeast of a35r3.

southeast of a28r3 is northwest of a36r3.

northeast of a10r6 is southwest of a37r6.

northeast of a38r1 is southwest of a39r1.

southwest of a40r0 is northeast of a41r0.

northwest of a14r4 is southeast of a42r4.

northeast of a37r6 is southwest of a43r6.

southeast of a19r7 is northwest of a44r7.

northeast of a21r6 is southwest of a45r6.

southeast of a22r6 is northwest of a46r6.

northwest of a5r0 is southeast of a47r0.

northwest of a7r6 is southeast of a48r6.

northwest of a9r2 is southeast of a49r2.

north of a11r2 is south of a50r2.

northeast of a13r4 is southwest of a51r4.

southeast of a2r8 is northwest of a52r8.

southeast of a30r7 is northwest of a53r7.

southwest of a31r7 is northeast of a54r7.

southeast of a32r1 is northwest of a55r1.

southeast of a33r5 is northwest of a56r5.

southwest of a34r5 is northeast of a57r5.

southwest of a35r3 is northeast of a58r3.

southeast of a36r3 is northwest of a59r3.

southwest of a0r8 is northeast of a60r8.

east of a24r5 is west of a26r5.

east of a25r3 is west of a28r3.

southwest of a29r1 is northeast of a39r1.

northeast of a6r0 is southwest of a41r0.

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
the player is in a55r1.

the orientation of the player is 6.
