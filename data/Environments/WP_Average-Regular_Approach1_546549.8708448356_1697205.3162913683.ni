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

[create a10r0]
a10r0 is a area. "An area (10) in r0".
the printed name of the a10r0 is "Room 0".
Understand "Area 10 in Room 0" as a10r0.
the parent of the a10r0 is r0.
the X of the a10r0 is "545890.0086022713".
the Y of the a10r0 is "1695564.0104014394".

[create a20r0]
a20r0 is a area. "An area (20) in r0".
the printed name of the a20r0 is "Room 0".
Understand "Area 20 in Room 0" as a20r0.
the parent of the a20r0 is r0.
the X of the a20r0 is "545786.8816727516".
the Y of the a20r0 is "1695419.4952958587".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "545903.7496886398".
the Y of the a22r0 is "1695562.3173736061".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "545997.2968536067".
the Y of the a23r0 is "1695451.4209743896".

[create a58r0]
a58r0 is a area. "An area (58) in r0".
the printed name of the a58r0 is "Room 0".
Understand "Area 58 in Room 0" as a58r0.
the parent of the a58r0 is r0.
the X of the a58r0 is "545756.0".
the Y of the a58r0 is "1695372.0".

[create a60r0]
a60r0 is a area. "An area (60) in r0".
the printed name of the a60r0 is "Room 0".
Understand "Area 60 in Room 0" as a60r0.
the parent of the a60r0 is r0.
the X of the a60r0 is "546038.0".
the Y of the a60r0 is "1695400.0".

[create a2r1]
a2r1 is a area. "An area (2) in r1".
the printed name of the a2r1 is "Room 1".
Understand "Area 2 in Room 1" as a2r1.
the parent of the a2r1 is r1.
the X of the a2r1 is "546477.3333333333".
the Y of the a2r1 is "1697460.9694444444".

[create a3r1]
a3r1 is a area. "An area (3) in r1".
the printed name of the a3r1 is "Room 1".
Understand "Area 3 in Room 1" as a3r1.
the parent of the a3r1 is r1.
the X of the a3r1 is "546544.5167685037".
the Y of the a3r1 is "1697373.5133957593".

[create a11r1]
a11r1 is a area. "An area (11) in r1".
the printed name of the a11r1 is "Room 1".
Understand "Area 11 in Room 1" as a11r1.
the parent of the a11r1 is r1.
the X of the a11r1 is "546549.8708448356".
the Y of the a11r1 is "1697205.3162913683".

[create a30r1]
a30r1 is a area. "An area (30) in r1".
the printed name of the a30r1 is "Room 1".
Understand "Area 30 in Room 1" as a30r1.
the parent of the a30r1 is r1.
the X of the a30r1 is "546644.0".
the Y of the a30r1 is "1697504.0".

[create a31r1]
a31r1 is a area. "An area (31) in r1".
the printed name of the a31r1 is "Room 1".
Understand "Area 31 in Room 1" as a31r1.
the parent of the a31r1 is r1.
the X of the a31r1 is "546610.6666666666".
the Y of the a31r1 is "1697453.5364495006".

[create a62r1]
a62r1 is a area. "An area (62) in r1".
the printed name of the a62r1 is "Room 1".
Understand "Area 62 in Room 1" as a62r1.
the parent of the a62r1 is r1.
the X of the a62r1 is "546444.0".
the Y of the a62r1 is "1697504.0".

[create a12r2]
a12r2 is a area. "An area (12) in r2".
the printed name of the a12r2 is "Room 2".
Understand "Area 12 in Room 2" as a12r2.
the parent of the a12r2 is r2.
the X of the a12r2 is "547144.2390740267".
the Y of the a12r2 is "1695562.4312827692".

[create a21r2]
a21r2 is a area. "An area (21) in r2".
the printed name of the a21r2 is "Room 2".
Understand "Area 21 in Room 2" as a21r2.
the parent of the a21r2 is r2.
the X of the a21r2 is "547048.7043713275".
the Y of the a21r2 is "1695422.7067112252".

[create a24r2]
a24r2 is a area. "An area (24) in r2".
the printed name of the a24r2 is "Room 2".
Understand "Area 24 in Room 2" as a24r2.
the parent of the a24r2 is r2.
the X of the a24r2 is "547166.2896516462".
the Y of the a24r2 is "1695560.8833195833".

[create a25r2]
a25r2 is a area. "An area (25) in r2".
the printed name of the a25r2 is "Room 2".
Understand "Area 25 in Room 2" as a25r2.
the parent of the a25r2 is r2.
the X of the a25r2 is "547260.4539705468".
the Y of the a25r2 is "1695448.7643334023".

[create a59r2]
a59r2 is a area. "An area (59) in r2".
the printed name of the a59r2 is "Room 2".
Understand "Area 59 in Room 2" as a59r2.
the parent of the a59r2 is r2.
the X of the a59r2 is "547018.0".
the Y of the a59r2 is "1695372.0".

[create a61r2]
a61r2 is a area. "An area (61) in r2".
the printed name of the a61r2 is "Room 2".
Understand "Area 61 in Room 2" as a61r2.
the parent of the a61r2 is r2.
the X of the a61r2 is "547301.0".
the Y of the a61r2 is "1695400.0".

[create a0r3]
a0r3 is a area. "An area (0) in r3".
the printed name of the a0r3 is "Room 3".
Understand "Area 0 in Room 3" as a0r3.
the parent of the a0r3 is r3.
the X of the a0r3 is "547342.4921500895".
the Y of the a0r3 is "1697460.7986347184".

[create a1r3]
a1r3 is a area. "An area (1) in r3".
the printed name of the a1r3 is "Room 3".
Understand "Area 1 in Room 3" as a1r3.
the parent of the a1r3 is r3.
the X of the a1r3 is "547630.4944995975".
the Y of the a1r3 is "1697411.0205706113".

[create a7r3]
a7r3 is a area. "An area (7) in r3".
the printed name of the a7r3 is "Room 3".
Understand "Area 7 in Room 3" as a7r3.
the parent of the a7r3 is r3.
the X of the a7r3 is "547677.1587808633".
the Y of the a7r3 is "1697381.3772771838".

[create a8r3]
a8r3 is a area. "An area (8) in r3".
the printed name of the a8r3 is "Room 3".
Understand "Area 8 in Room 3" as a8r3.
the parent of the a8r3 is r3.
the X of the a8r3 is "548000.4450183739".
the Y of the a8r3 is "1697335.9641000088".

[create a9r3]
a9r3 is a area. "An area (9) in r3".
the printed name of the a9r3 is "Room 3".
Understand "Area 9 in Room 3" as a9r3.
the parent of the a9r3 is r3.
the X of the a9r3 is "548005.2384354565".
the Y of the a9r3 is "1697340.590255913".

[create a15r3]
a15r3 is a area. "An area (15) in r3".
the printed name of the a15r3 is "Room 3".
Understand "Area 15 in Room 3" as a15r3.
the parent of the a15r3 is r3.
the X of the a15r3 is "549012.7894678493".
the Y of the a15r3 is "1697112.9604600752".

[create a16r3]
a16r3 is a area. "An area (16) in r3".
the printed name of the a16r3 is "Room 3".
Understand "Area 16 in Room 3" as a16r3.
the parent of the a16r3 is r3.
the X of the a16r3 is "549013.3229297197".
the Y of the a16r3 is "1696741.7692602102".

[create a17r3]
a17r3 is a area. "An area (17) in r3".
the printed name of the a17r3 is "Room 3".
Understand "Area 17 in Room 3" as a17r3.
the parent of the a17r3 is r3.
the X of the a17r3 is "548761.5663719162".
the Y of the a17r3 is "1697391.0382126388".

[create a34r3]
a34r3 is a area. "An area (34) in r3".
the printed name of the a34r3 is "Room 3".
Understand "Area 34 in Room 3" as a34r3.
the parent of the a34r3 is r3.
the X of the a34r3 is "547295.0".
the Y of the a34r3 is "1697514.0".

[create a35r3]
a35r3 is a area. "An area (35) in r3".
the printed name of the a35r3 is "Room 3".
Understand "Area 35 in Room 3" as a35r3.
the parent of the a35r3 is r3.
the X of the a35r3 is "547325.915378007".
the Y of the a35r3 is "1697488.1834621993".

[create a38r3]
a38r3 is a area. "An area (38) in r3".
the printed name of the a38r3 is "Room 3".
Understand "Area 38 in Room 3" as a38r3.
the parent of the a38r3 is r3.
the X of the a38r3 is "547589.0".
the Y of the a38r3 is "1697514.0".

[create a39r3]
a39r3 is a area. "An area (39) in r3".
the printed name of the a39r3 is "Room 3".
Understand "Area 39 in Room 3" as a39r3.
the parent of the a39r3 is r3.
the X of the a39r3 is "547620.826460481".
the Y of the a39r3 is "1697505.767353952".

[create a40r3]
a40r3 is a area. "An area (40) in r3".
the printed name of the a40r3 is "Room 3".
Understand "Area 40 in Room 3" as a40r3.
the parent of the a40r3 is r3.
the X of the a40r3 is "548054.0".
the Y of the a40r3 is "1697514.0".

[create a41r3]
a41r3 is a area. "An area (41) in r3".
the printed name of the a41r3 is "Room 3".
Understand "Area 41 in Room 3" as a41r3.
the parent of the a41r3 is r3.
the X of the a41r3 is "548022.005982906".
the Y of the a41r3 is "1697505.7505982907".

[create a50r3]
a50r3 is a area. "An area (50) in r3".
the printed name of the a50r3 is "Room 3".
Understand "Area 50 in Room 3" as a50r3.
the parent of the a50r3 is r3.
the X of the a50r3 is "549162.0".
the Y of the a50r3 is "1696678.0".

[create a51r3]
a51r3 is a area. "An area (51) in r3".
the printed name of the a51r3 is "Room 3".
Understand "Area 51 in Room 3" as a51r3.
the parent of the a51r3 is r3.
the X of the a51r3 is "549075.8956183488".
the Y of the a51r3 is "1696713.7857142857".

[create a52r3]
a52r3 is a area. "An area (52) in r3".
the printed name of the a52r3 is "Room 3".
Understand "Area 52 in Room 3" as a52r3.
the parent of the a52r3 is r3.
the X of the a52r3 is "549162.0".
the Y of the a52r3 is "1697179.0".

[create a53r3]
a53r3 is a area. "An area (53) in r3".
the printed name of the a53r3 is "Room 3".
Understand "Area 53 in Room 3" as a53r3.
the parent of the a53r3 is r3.
the X of the a53r3 is "549082.7885154061".
the Y of the a53r3 is "1697143.2142857143".

[create a54r3]
a54r3 is a area. "An area (54) in r3".
the printed name of the a54r3 is "Room 3".
Understand "Area 54 in Room 3" as a54r3.
the parent of the a54r3 is r3.
the X of the a54r3 is "548805.0".
the Y of the a54r3 is "1697504.0".

[create a55r3]
a55r3 is a area. "An area (55) in r3".
the printed name of the a55r3 is "Room 3".
Understand "Area 55 in Room 3" as a55r3.
the parent of the a55r3 is r3.
the X of the a55r3 is "548771.2".
the Y of the a55r3 is "1697415.6469450549".

[create a56r3]
a56r3 is a area. "An area (56) in r3".
the printed name of the a56r3 is "Room 3".
Understand "Area 56 in Room 3" as a56r3.
the parent of the a56r3 is r3.
the X of the a56r3 is "547286.0".
the Y of the a56r3 is "1697424.0".

[create a57r3]
a57r3 is a area. "An area (57) in r3".
the printed name of the a57r3 is "Room 3".
Understand "Area 57 in Room 3" as a57r3.
the parent of the a57r3 is r3.
the X of the a57r3 is "547326.7622699386".
the Y of the a57r3 is "1697442.6487730062".

[create a13r4]
a13r4 is a area. "An area (13) in r4".
the printed name of the a13r4 is "Room 4".
Understand "Area 13 in Room 4" as a13r4.
the parent of the a13r4 is r4.
the X of the a13r4 is "548000.7860830325".
the Y of the a13r4 is "1696338.5".

[create a14r4]
a14r4 is a area. "An area (14) in r4".
the printed name of the a14r4 is "Room 4".
Understand "Area 14 in Room 4" as a14r4.
the parent of the a14r4 is r4.
the X of the a14r4 is "547994.0".
the Y of the a14r4 is "1695568.577830189".

[create a18r4]
a18r4 is a area. "An area (18) in r4".
the printed name of the a18r4 is "Room 4".
Understand "Area 18 in Room 4" as a18r4.
the parent of the a18r4 is r4.
the X of the a18r4 is "548360.7817693062".
the Y of the a18r4 is "1695444.9748289345".

[create a19r4]
a19r4 is a area. "An area (19) in r4".
the printed name of the a19r4 is "Room 4".
Understand "Area 19 in Room 4" as a19r4.
the parent of the a19r4 is r4.
the X of the a19r4 is "547612.389199255".
the Y of the a19r4 is "1695440.2597765361".

[create a42r4]
a42r4 is a area. "An area (42) in r4".
the printed name of the a42r4 is "Room 4".
Understand "Area 42 in Room 4" as a42r4.
the parent of the a42r4 is r4.
the X of the a42r4 is "547567.0".
the Y of the a42r4 is "1695391.0".

[create a43r4]
a43r4 is a area. "An area (43) in r4".
the printed name of the a43r4 is "Room 4".
Understand "Area 43 in Room 4" as a43r4.
the parent of the a43r4 is r4.
the X of the a43r4 is "547598.2137161084".
the Y of the a43r4 is "1695416.633971292".

[create a44r4]
a44r4 is a area. "An area (44) in r4".
the printed name of the a44r4 is "Room 4".
Understand "Area 44 in Room 4" as a44r4.
the parent of the a44r4 is r4.
the X of the a44r4 is "547558.0".
the Y of the a44r4 is "1695480.0".

[create a45r4]
a45r4 is a area. "An area (45) in r4".
the printed name of the a45r4 is "Room 4".
Understand "Area 45 in Room 4" as a45r4.
the parent of the a45r4 is r4.
the X of the a45r4 is "547589.0059757691".
the Y of the a45r4 is "1695460.6579076622".

[create a46r4]
a46r4 is a area. "An area (46) in r4".
the printed name of the a46r4 is "Room 4".
Understand "Area 46 in Room 4" as a46r4.
the parent of the a46r4 is r4.
the X of the a46r4 is "548409.0".
the Y of the a46r4 is "1695391.0".

[create a47r4]
a47r4 is a area. "An area (47) in r4".
the printed name of the a47r4 is "Room 4".
Understand "Area 47 in Room 4" as a47r4.
the parent of the a47r4 is r4.
the X of the a47r4 is "548377.7862838916".
the Y of the a47r4 is "1695416.633971292".

[create a48r4]
a48r4 is a area. "An area (48) in r4".
the printed name of the a48r4 is "Room 4".
Understand "Area 48 in Room 4" as a48r4.
the parent of the a48r4 is r4.
the X of the a48r4 is "548418.0".
the Y of the a48r4 is "1695480.0".

[create a49r4]
a49r4 is a area. "An area (49) in r4".
the printed name of the a49r4 is "Room 4".
Understand "Area 49 in Room 4" as a49r4.
the parent of the a49r4 is r4.
the X of the a49r4 is "548375.1485252038".
the Y of the a49r4 is "1695461.8557671143".

[create a4r5]
a4r5 is a area. "An area (4) in r5".
the printed name of the a4r5 is "Room 5".
Understand "Area 4 in Room 5" as a4r5.
the parent of the a4r5 is r5.
the X of the a4r5 is "545050.2813718208".
the Y of the a4r5 is "1697461.2845894436".

[create a5r5]
a5r5 is a area. "An area (5) in r5".
the printed name of the a5r5 is "Room 5".
Understand "Area 5 in Room 5" as a5r5.
the parent of the a5r5 is r5.
the X of the a5r5 is "545405.79770735".
the Y of the a5r5 is "1697344.5677680378".

[create a6r5]
a6r5 is a area. "An area (6) in r5".
the printed name of the a6r5 is "Room 5".
Understand "Area 6 in Room 5" as a6r5.
the parent of the a6r5 is r5.
the X of the a6r5 is "545779.1653756485".
the Y of the a6r5 is "1697461.2658517451".

[create a26r5]
a26r5 is a area. "An area (26) in r5".
the printed name of the a26r5 is "Room 5".
Understand "Area 26 in Room 5" as a26r5.
the parent of the a26r5 is r5.
the X of the a26r5 is "544994.0".
the Y of the a26r5 is "1697424.0".

[create a27r5]
a27r5 is a area. "An area (27) in r5".
the printed name of the a27r5 is "Room 5".
Understand "Area 27 in Room 5" as a27r5.
the parent of the a27r5 is r5.
the X of the a27r5 is "545033.4941364606".
the Y of the a27r5 is "1697442.775586354".

[create a28r5]
a28r5 is a area. "An area (28) in r5".
the printed name of the a28r5 is "Room 5".
Understand "Area 28 in Room 5" as a28r5.
the parent of the a28r5 is r5.
the X of the a28r5 is "545836.0".
the Y of the a28r5 is "1697424.0".

[create a29r5]
a29r5 is a area. "An area (29) in r5".
the printed name of the a29r5 is "Room 5".
Understand "Area 29 in Room 5" as a29r5.
the parent of the a29r5 is r5.
the X of the a29r5 is "545795.6686930091".
the Y of the a29r5 is "1697442.2965214455".

[create a32r5]
a32r5 is a area. "An area (32) in r5".
the printed name of the a32r5 is "Room 5".
Understand "Area 32 in Room 5" as a32r5.
the parent of the a32r5 is r5.
the X of the a32r5 is "545826.0".
the Y of the a32r5 is "1697514.0".

[create a33r5]
a33r5 is a area. "An area (33) in r5".
the printed name of the a33r5 is "Room 5".
Understand "Area 33 in Room 5" as a33r5.
the parent of the a33r5 is r5.
the X of the a33r5 is "545795.1050372066".
the Y of the a33r5 is "1697487.7894485786".

[create a36r5]
a36r5 is a area. "An area (36) in r5".
the printed name of the a36r5 is "Room 5".
Understand "Area 36 in Room 5" as a36r5.
the parent of the a36r5 is r5.
the X of the a36r5 is "545003.0".
the Y of the a36r5 is "1697514.0".

[create a37r5]
a37r5 is a area. "An area (37) in r5".
the printed name of the a37r5 is "Room 5".
Understand "Area 37 in Room 5" as a37r5.
the parent of the a37r5 is r5.
the X of the a37r5 is "545034.087606838".
the Y of the a37r5 is "1697488.166239316".

[create d0]
d0 is a door. "A door between a24r2 and a7r3".
d0 is north of a24r2 and south of a7r3.

[create d2]
d2 is a door. "A door between a22r0 and a11r1".
d2 is north of a22r0 and south of a11r1.

[create d4]
d4 is a door. "A door between a10r0 and a5r5".
d4 is north of a10r0 and south of a5r5.

[create d6]
d6 is a door. "A door between a11r1 and a12r2".
d6 is southeast of a11r1 and northwest of a12r2.

[create d8]
d8 is a door. "A door between a8r3 and a13r4".
d8 is south of a8r3 and north of a13r4.

[create d9]
d9 is a door. "A door between a16r3 and a13r4".
d9 is west of a16r3 and east of a13r4.

east of a0r3 is west of a1r3.

southeast of a2r1 is northwest of a3r1.

east of a4r5 is west of a5r5.

east of a5r5 is west of a6r5.

east of a7r3 is west of a8r3.

northeast of a8r3 is southwest of a9r3.

south of a13r4 is north of a14r4.

south of a15r3 is north of a16r3.

northwest of a15r3 is southeast of a17r3.

south of a3r1 is north of a11r1.

east of a9r3 is west of a17r3.

east of a14r4 is west of a18r4.

west of a14r4 is east of a19r4.

southwest of a10r0 is northeast of a20r0.

southwest of a12r2 is northeast of a21r2.

southeast of a22r0 is northwest of a23r0.

southeast of a24r2 is northwest of a25r2.

northeast of a26r5 is southwest of a27r5.

northwest of a28r5 is southeast of a29r5.

southwest of a30r1 is northeast of a31r1.

southwest of a32r5 is northeast of a33r5.

southeast of a34r3 is northwest of a35r3.

southeast of a36r5 is northwest of a37r5.

east of a38r3 is west of a39r3.

west of a40r3 is east of a41r3.

northeast of a42r4 is southwest of a43r4.

southeast of a44r4 is northwest of a45r4.

northwest of a46r4 is southeast of a47r4.

southwest of a48r4 is northeast of a49r4.

northwest of a50r3 is southeast of a51r3.

southwest of a52r3 is northeast of a53r3.

south of a54r3 is north of a55r3.

northeast of a56r3 is southwest of a57r3.

southwest of a20r0 is northeast of a58r0.

southwest of a21r2 is northeast of a59r2.

southeast of a23r0 is northwest of a60r0.

southeast of a25r2 is northwest of a61r2.

northwest of a2r1 is southeast of a62r1.

northwest of a0r3 is southeast of a35r3.

northwest of a4r5 is southeast of a37r5.

north of a1r3 is south of a39r3.

north of a9r3 is south of a41r3.

northeast of a15r3 is southwest of a53r3.

north of a17r3 is south of a55r3.

southeast of a1r3 is northwest of a7r3.

southwest of a0r3 is northeast of a57r3.

southwest of a4r5 is northeast of a27r5.

southeast of a6r5 is northwest of a29r5.

northeast of a3r1 is southwest of a31r1.

northeast of a6r5 is southwest of a33r5.

southwest of a19r4 is northeast of a43r4.

northwest of a19r4 is southeast of a45r4.

southeast of a18r4 is northwest of a47r4.

northeast of a18r4 is southwest of a49r4.

east of a12r2 is west of a24r2.

east of a10r0 is west of a22r0.

southeast of a16r3 is northwest of a51r3.

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
the player is in a11r1.

the orientation of the player is 6.
