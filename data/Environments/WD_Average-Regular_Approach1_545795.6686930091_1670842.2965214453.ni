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

[create a3r0]
a3r0 is a area. "An area (3) in r0".
the printed name of the a3r0 is "Room 0".
Understand "Area 3 in Room 0" as a3r0.
the parent of the a3r0 is r0.
the X of the a3r0 is "545786.8816727516".
the Y of the a3r0 is "1668819.4952958585".

[create a4r0]
a4r0 is a area. "An area (4) in r0".
the printed name of the a4r0 is "Room 0".
Understand "Area 4 in Room 0" as a4r0.
the parent of the a4r0 is r0.
the X of the a4r0 is "545890.0086022713".
the Y of the a4r0 is "1668964.0104014392".

[create a8r0]
a8r0 is a area. "An area (8) in r0".
the printed name of the a8r0 is "Room 0".
Understand "Area 8 in Room 0" as a8r0.
the parent of the a8r0 is r0.
the X of the a8r0 is "545903.7496886395".
the Y of the a8r0 is "1668962.317373606".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "545997.2968536066".
the Y of the a9r0 is "1668851.4209743899".

[create a57r0]
a57r0 is a area. "An area (57) in r0".
the printed name of the a57r0 is "Room 0".
Understand "Area 57 in Room 0" as a57r0.
the parent of the a57r0 is r0.
the X of the a57r0 is "545756.0".
the Y of the a57r0 is "1668772.0".

[create a59r0]
a59r0 is a area. "An area (59) in r0".
the printed name of the a59r0 is "Room 0".
Understand "Area 59 in Room 0" as a59r0.
the parent of the a59r0 is r0.
the X of the a59r0 is "546038.0".
the Y of the a59r0 is "1668800.0".

[create a12r1]
a12r1 is a area. "An area (12) in r1".
the printed name of the a12r1 is "Room 1".
Understand "Area 12 in Room 1" as a12r1.
the parent of the a12r1 is r1.
the X of the a12r1 is "546544.5167685037".
the Y of the a12r1 is "1670773.513395759".

[create a13r1]
a13r1 is a area. "An area (13) in r1".
the printed name of the a13r1 is "Room 1".
Understand "Area 13 in Room 1" as a13r1.
the parent of the a13r1 is r1.
the X of the a13r1 is "546549.8708448356".
the Y of the a13r1 is "1670605.3162913686".

[create a18r1]
a18r1 is a area. "An area (18) in r1".
the printed name of the a18r1 is "Room 1".
Understand "Area 18 in Room 1" as a18r1.
the parent of the a18r1 is r1.
the X of the a18r1 is "546477.3333333333".
the Y of the a18r1 is "1670860.9694444442".

[create a40r1]
a40r1 is a area. "An area (40) in r1".
the printed name of the a40r1 is "Room 1".
Understand "Area 40 in Room 1" as a40r1.
the parent of the a40r1 is r1.
the X of the a40r1 is "546644.0".
the Y of the a40r1 is "1670904.0".

[create a41r1]
a41r1 is a area. "An area (41) in r1".
the printed name of the a41r1 is "Room 1".
Understand "Area 41 in Room 1" as a41r1.
the parent of the a41r1 is r1.
the X of the a41r1 is "546610.6666666666".
the Y of the a41r1 is "1670853.5364495006".

[create a56r1]
a56r1 is a area. "An area (56) in r1".
the printed name of the a56r1 is "Room 1".
Understand "Area 56 in Room 1" as a56r1.
the parent of the a56r1 is r1.
the X of the a56r1 is "546444.0".
the Y of the a56r1 is "1670904.0".

[create a5r2]
a5r2 is a area. "An area (5) in r2".
the printed name of the a5r2 is "Room 2".
Understand "Area 5 in Room 2" as a5r2.
the parent of the a5r2 is r2.
the X of the a5r2 is "547048.7043713275".
the Y of the a5r2 is "1668822.7067112252".

[create a6r2]
a6r2 is a area. "An area (6) in r2".
the printed name of the a6r2 is "Room 2".
Understand "Area 6 in Room 2" as a6r2.
the parent of the a6r2 is r2.
the X of the a6r2 is "547144.2390740267".
the Y of the a6r2 is "1668962.431282769".

[create a24r2]
a24r2 is a area. "An area (24) in r2".
the printed name of the a24r2 is "Room 2".
Understand "Area 24 in Room 2" as a24r2.
the parent of the a24r2 is r2.
the X of the a24r2 is "547166.2896516462".
the Y of the a24r2 is "1668960.8833195833".

[create a25r2]
a25r2 is a area. "An area (25) in r2".
the printed name of the a25r2 is "Room 2".
Understand "Area 25 in Room 2" as a25r2.
the parent of the a25r2 is r2.
the X of the a25r2 is "547260.4539705468".
the Y of the a25r2 is "1668848.7643334025".

[create a58r2]
a58r2 is a area. "An area (58) in r2".
the printed name of the a58r2 is "Room 2".
Understand "Area 58 in Room 2" as a58r2.
the parent of the a58r2 is r2.
the X of the a58r2 is "547018.0".
the Y of the a58r2 is "1668772.0".

[create a60r2]
a60r2 is a area. "An area (60) in r2".
the printed name of the a60r2 is "Room 2".
Understand "Area 60 in Room 2" as a60r2.
the parent of the a60r2 is r2.
the X of the a60r2 is "547301.0".
the Y of the a60r2 is "1668800.0".

[create a14r3]
a14r3 is a area. "An area (14) in r3".
the printed name of the a14r3 is "Room 3".
Understand "Area 14 in Room 3" as a14r3.
the parent of the a14r3 is r3.
the X of the a14r3 is "548005.9253140049".
the Y of the a14r3 is "1670747.3560096093".

[create a15r3]
a15r3 is a area. "An area (15) in r3".
the printed name of the a15r3 is "Room 3".
Understand "Area 15 in Room 3" as a15r3.
the parent of the a15r3 is r3.
the X of the a15r3 is "549128.03553441".
the Y of the a15r3 is "1670783.5963231502".

[create a16r3]
a16r3 is a area. "An area (16) in r3".
the printed name of the a16r3 is "Room 3".
Understand "Area 16 in Room 3" as a16r3.
the parent of the a16r3 is r3.
the X of the a16r3 is "547342.4921500895".
the Y of the a16r3 is "1670860.7986347184".

[create a17r3]
a17r3 is a area. "An area (17) in r3".
the printed name of the a17r3 is "Room 3".
Understand "Area 17 in Room 3" as a17r3.
the parent of the a17r3 is r3.
the X of the a17r3 is "547630.4944995975".
the Y of the a17r3 is "1670811.0205706109".

[create a22r3]
a22r3 is a area. "An area (22) in r3".
the printed name of the a22r3 is "Room 3".
Understand "Area 22 in Room 3" as a22r3.
the parent of the a22r3 is r3.
the X of the a22r3 is "547677.1587808633".
the Y of the a22r3 is "1670781.3772771838".

[create a23r3]
a23r3 is a area. "An area (23) in r3".
the printed name of the a23r3 is "Room 3".
Understand "Area 23 in Room 3" as a23r3.
the parent of the a23r3 is r3.
the X of the a23r3 is "547996.7084964575".
the Y of the a23r3 is "1670739.2008955854".

[create a28r3]
a28r3 is a area. "An area (28) in r3".
the printed name of the a28r3 is "Room 3".
Understand "Area 28 in Room 3" as a28r3.
the parent of the a28r3 is r3.
the X of the a28r3 is "547589.0".
the Y of the a28r3 is "1670914.0".

[create a29r3]
a29r3 is a area. "An area (29) in r3".
the printed name of the a29r3 is "Room 3".
Understand "Area 29 in Room 3" as a29r3.
the parent of the a29r3 is r3.
the X of the a29r3 is "547620.8264604809".
the Y of the a29r3 is "1670905.767353952".

[create a30r3]
a30r3 is a area. "An area (30) in r3".
the printed name of the a30r3 is "Room 3".
Understand "Area 30 in Room 3" as a30r3.
the parent of the a30r3 is r3.
the X of the a30r3 is "548054.0".
the Y of the a30r3 is "1670914.0".

[create a31r3]
a31r3 is a area. "An area (31) in r3".
the printed name of the a31r3 is "Room 3".
Understand "Area 31 in Room 3" as a31r3.
the parent of the a31r3 is r3.
the X of the a31r3 is "548022.005982906".
the Y of the a31r3 is "1670905.7505982907".

[create a34r3]
a34r3 is a area. "An area (34) in r3".
the printed name of the a34r3 is "Room 3".
Understand "Area 34 in Room 3" as a34r3.
the parent of the a34r3 is r3.
the X of the a34r3 is "547286.0".
the Y of the a34r3 is "1670824.0".

[create a35r3]
a35r3 is a area. "An area (35) in r3".
the printed name of the a35r3 is "Room 3".
Understand "Area 35 in Room 3" as a35r3.
the parent of the a35r3 is r3.
the X of the a35r3 is "547326.7622699386".
the Y of the a35r3 is "1670842.6487730062".

[create a44r3]
a44r3 is a area. "An area (44) in r3".
the printed name of the a44r3 is "Room 3".
Understand "Area 44 in Room 3" as a44r3.
the parent of the a44r3 is r3.
the X of the a44r3 is "547295.0".
the Y of the a44r3 is "1670914.0".

[create a45r3]
a45r3 is a area. "An area (45) in r3".
the printed name of the a45r3 is "Room 3".
Understand "Area 45 in Room 3" as a45r3.
the parent of the a45r3 is r3.
the X of the a45r3 is "547325.915378007".
the Y of the a45r3 is "1670888.1834621993".

[create a63r3]
a63r3 is a area. "An area (63) in r3".
the printed name of the a63r3 is "Room 3".
Understand "Area 63 in Room 3" as a63r3.
the parent of the a63r3 is r3.
the X of the a63r3 is "549164.8043478262".
the Y of the a63r3 is "1670775.0".

[create a0r4]
a0r4 is a area. "An area (0) in r4".
the printed name of the a0r4 is "Room 4".
Understand "Area 0 in Room 4" as a0r4.
the parent of the a0r4 is r4.
the X of the a0r4 is "547612.389199255".
the Y of the a0r4 is "1668840.2597765361".

[create a1r4]
a1r4 is a area. "An area (1) in r4".
the printed name of the a1r4 is "Room 4".
Understand "Area 1 in Room 4" as a1r4.
the parent of the a1r4 is r4.
the X of the a1r4 is "547996.1865194305".
the Y of the a1r4 is "1668970.3208574706".

[create a2r4]
a2r4 is a area. "An area (2) in r4".
the printed name of the a2r4 is "Room 4".
Understand "Area 2 in Room 4" as a2r4.
the parent of the a2r4 is r4.
the X of the a2r4 is "549015.4184328842".
the Y of the a2r4 is "1668936.1969348686".

[create a7r4]
a7r4 is a area. "An area (7) in r4".
the printed name of the a7r4 is "Room 4".
Understand "Area 7 in Room 4" as a7r4.
the parent of the a7r4 is r4.
the X of the a7r4 is "549496.0324175665".
the Y of the a7r4 is "1669502.0135168612".

[create a48r4]
a48r4 is a area. "An area (48) in r4".
the printed name of the a48r4 is "Room 4".
Understand "Area 48 in Room 4" as a48r4.
the parent of the a48r4 is r4.
the X of the a48r4 is "547567.0".
the Y of the a48r4 is "1668791.0".

[create a49r4]
a49r4 is a area. "An area (49) in r4".
the printed name of the a49r4 is "Room 4".
Understand "Area 49 in Room 4" as a49r4.
the parent of the a49r4 is r4.
the X of the a49r4 is "547598.2137161084".
the Y of the a49r4 is "1668816.633971292".

[create a50r4]
a50r4 is a area. "An area (50) in r4".
the printed name of the a50r4 is "Room 4".
Understand "Area 50 in Room 4" as a50r4.
the parent of the a50r4 is r4.
the X of the a50r4 is "547558.0".
the Y of the a50r4 is "1668880.0".

[create a51r4]
a51r4 is a area. "An area (51) in r4".
the printed name of the a51r4 is "Room 4".
Understand "Area 51 in Room 4" as a51r4.
the parent of the a51r4 is r4.
the X of the a51r4 is "547589.0059757691".
the Y of the a51r4 is "1668860.657907662".

[create a52r4]
a52r4 is a area. "An area (52) in r4".
the printed name of the a52r4 is "Room 4".
Understand "Area 52 in Room 4" as a52r4.
the parent of the a52r4 is r4.
the X of the a52r4 is "549073.0".
the Y of the a52r4 is "1668800.0".

[create a53r4]
a53r4 is a area. "An area (53) in r4".
the printed name of the a53r4 is "Room 4".
Understand "Area 53 in Room 4" as a53r4.
the parent of the a53r4 is r4.
the X of the a53r4 is "549036.875".
the Y of the a53r4 is "1668885.7209595959".

[create a54r4]
a54r4 is a area. "An area (54) in r4".
the printed name of the a54r4 is "Room 4".
Understand "Area 54 in Room 4" as a54r4.
the parent of the a54r4 is r4.
the X of the a54r4 is "549681.0".
the Y of the a54r4 is "1669427.0".

[create a55r4]
a55r4 is a area. "An area (55) in r4".
the printed name of the a55r4 is "Room 4".
Understand "Area 55 in Room 4" as a55r4.
the parent of the a55r4 is r4.
the X of the a55r4 is "549592.3368675594".
the Y of the a55r4 is "1669462.2857142857".

[create a26r5]
a26r5 is a area. "An area (26) in r5".
the printed name of the a26r5 is "Room 5".
Understand "Area 26 in Room 5" as a26r5.
the parent of the a26r5 is r5.
the X of the a26r5 is "549380.4145304462".
the Y of the a26r5 is "1670638.8846710813".

[create a61r5]
a61r5 is a area. "An area (61) in r5".
the printed name of the a61r5 is "Room 5".
Understand "Area 61 in Room 5" as a61r5.
the parent of the a61r5 is r5.
the X of the a61r5 is "549358.0".
the Y of the a61r5 is "1670675.677419355".

[create a19r6]
a19r6 is a area. "An area (19) in r6".
the printed name of the a19r6 is "Room 6".
Understand "Area 19 in Room 6" as a19r6.
the parent of the a19r6 is r6.
the X of the a19r6 is "545050.2813718207".
the Y of the a19r6 is "1670861.2845894434".

[create a20r6]
a20r6 is a area. "An area (20) in r6".
the printed name of the a20r6 is "Room 6".
Understand "Area 20 in Room 6" as a20r6.
the parent of the a20r6 is r6.
the X of the a20r6 is "545405.79770735".
the Y of the a20r6 is "1670744.567768038".

[create a21r6]
a21r6 is a area. "An area (21) in r6".
the printed name of the a21r6 is "Room 6".
Understand "Area 21 in Room 6" as a21r6.
the parent of the a21r6 is r6.
the X of the a21r6 is "545779.1653756485".
the Y of the a21r6 is "1670861.2658517451".

[create a36r6]
a36r6 is a area. "An area (36) in r6".
the printed name of the a36r6 is "Room 6".
Understand "Area 36 in Room 6" as a36r6.
the parent of the a36r6 is r6.
the X of the a36r6 is "544994.0".
the Y of the a36r6 is "1670824.0".

[create a37r6]
a37r6 is a area. "An area (37) in r6".
the printed name of the a37r6 is "Room 6".
Understand "Area 37 in Room 6" as a37r6.
the parent of the a37r6 is r6.
the X of the a37r6 is "545033.4941364606".
the Y of the a37r6 is "1670842.775586354".

[create a38r6]
a38r6 is a area. "An area (38) in r6".
the printed name of the a38r6 is "Room 6".
Understand "Area 38 in Room 6" as a38r6.
the parent of the a38r6 is r6.
the X of the a38r6 is "545836.0".
the Y of the a38r6 is "1670824.0".

[create a39r6]
a39r6 is a area. "An area (39) in r6".
the printed name of the a39r6 is "Room 6".
Understand "Area 39 in Room 6" as a39r6.
the parent of the a39r6 is r6.
the X of the a39r6 is "545795.6686930091".
the Y of the a39r6 is "1670842.2965214453".

[create a42r6]
a42r6 is a area. "An area (42) in r6".
the printed name of the a42r6 is "Room 6".
Understand "Area 42 in Room 6" as a42r6.
the parent of the a42r6 is r6.
the X of the a42r6 is "545826.0".
the Y of the a42r6 is "1670914.0".

[create a43r6]
a43r6 is a area. "An area (43) in r6".
the printed name of the a43r6 is "Room 6".
Understand "Area 43 in Room 6" as a43r6.
the parent of the a43r6 is r6.
the X of the a43r6 is "545795.1050372066".
the Y of the a43r6 is "1670887.7894485786".

[create a46r6]
a46r6 is a area. "An area (46) in r6".
the printed name of the a46r6 is "Room 6".
Understand "Area 46 in Room 6" as a46r6.
the parent of the a46r6 is r6.
the X of the a46r6 is "545003.0".
the Y of the a46r6 is "1670914.0".

[create a47r6]
a47r6 is a area. "An area (47) in r6".
the printed name of the a47r6 is "Room 6".
Understand "Area 47 in Room 6" as a47r6.
the parent of the a47r6 is r6.
the X of the a47r6 is "545034.0876068379".
the Y of the a47r6 is "1670888.1662393163".

[create a10r7]
a10r7 is a area. "An area (10) in r7".
the printed name of the a10r7 is "Room 7".
Understand "Area 10 in Room 7" as a10r7.
the parent of the a10r7 is r7.
the X of the a10r7 is "549261.402173913".
the Y of the a10r7 is "1670763.5525401558".

[create a11r7]
a11r7 is a area. "An area (11) in r7".
the printed name of the a11r7 is "Room 7".
Understand "Area 11 in Room 7" as a11r7.
the parent of the a11r7 is r7.
the X of the a11r7 is "549319.8286071356".
the Y of the a11r7 is "1670742.424978698".

[create a27r7]
a27r7 is a area. "An area (27) in r7".
the printed name of the a27r7 is "Room 7".
Understand "Area 27 in Room 7" as a27r7.
the parent of the a27r7 is r7.
the X of the a27r7 is "549342.6302481609".
the Y of the a27r7 is "1670700.9063846373".

[create a32r7]
a32r7 is a area. "An area (32) in r7".
the printed name of the a32r7 is "Room 7".
Understand "Area 32 in Room 7" as a32r7.
the parent of the a32r7 is r7.
the X of the a32r7 is "549358.0".
the Y of the a32r7 is "1670775.0".

[create a33r7]
a33r7 is a area. "An area (33) in r7".
the printed name of the a33r7 is "Room 7".
Understand "Area 33 in Room 7" as a33r7.
the parent of the a33r7 is r7.
the X of the a33r7 is "549325.8007246378".
the Y of the a33r7 is "1670750.1693548388".

[create a61r7]
a61r7 is a area. "An area (61) in r7".
the printed name of the a61r7 is "Room 7".
Understand "Area 61 in Room 7" as a61r7.
the parent of the a61r7 is r7.
the X of the a61r7 is "549358.0".
the Y of the a61r7 is "1670675.677419355".

[create a62r7]
a62r7 is a area. "An area (62) in r7".
the printed name of the a62r7 is "Room 7".
Understand "Area 62 in Room 7" as a62r7.
the parent of the a62r7 is r7.
the X of the a62r7 is "549183.0".
the Y of the a62r7 is "1670766.0".

[create a63r7]
a63r7 is a area. "An area (63) in r7".
the printed name of the a63r7 is "Room 7".
Understand "Area 63 in Room 7" as a63r7.
the parent of the a63r7 is r7.
the X of the a63r7 is "549164.8043478262".
the Y of the a63r7 is "1670775.0".

[create d0]
d0 is a door. "A door between a24r2 and a22r3".
d0 is north of a24r2 and south of a22r3.

[create d2]
d2 is a door. "A door between a8r0 and a13r1".
d2 is north of a8r0 and south of a13r1.

[create d4]
d4 is a door. "A door between a4r0 and a20r6".
d4 is north of a4r0 and south of a20r6.

[create d6]
d6 is a door. "A door between a13r1 and a6r2".
d6 is southeast of a13r1 and northwest of a6r2.

[create d8]
d8 is a door. "A door between a23r3 and a1r4".
d8 is south of a23r3 and north of a1r4.

[create d10]
d10 is a door. "A door between a7r4 and a26r5".
d10 is north of a7r4 and south of a26r5.

east of a0r4 is west of a1r4.

east of a1r4 is west of a2r4.

northeast of a3r0 is southwest of a4r0.

northeast of a5r2 is southwest of a6r2.

northeast of a2r4 is southwest of a7r4.

southeast of a8r0 is northwest of a9r0.

east of a10r7 is west of a11r7.

south of a12r1 is north of a13r1.

east of a14r3 is west of a15r3.

east of a16r3 is west of a17r3.

northwest of a12r1 is southeast of a18r1.

east of a19r6 is west of a20r6.

east of a20r6 is west of a21r6.

east of a22r3 is west of a23r3.

southeast of a24r2 is northwest of a25r2.

southeast of a11r7 is northwest of a27r7.

east of a28r3 is west of a29r3.

west of a30r3 is east of a31r3.

southwest of a32r7 is northeast of a33r7.

northeast of a34r3 is southwest of a35r3.

northeast of a36r6 is southwest of a37r6.

northwest of a38r6 is southeast of a39r6.

southwest of a40r1 is northeast of a41r1.

southwest of a42r6 is northeast of a43r6.

southeast of a44r3 is northwest of a45r3.

southeast of a46r6 is northwest of a47r6.

northeast of a48r4 is southwest of a49r4.

southeast of a50r4 is northwest of a51r4.

northwest of a52r4 is southeast of a53r4.

west of a54r4 is east of a55r4.

northwest of a18r1 is southeast of a56r1.

southwest of a3r0 is northeast of a57r0.

southwest of a5r2 is northeast of a58r2.

southeast of a9r0 is northwest of a59r0.

southeast of a25r2 is northwest of a60r2.

northwest of a26r5 is southeast of a61r5.

northwest of a61r5 is southeast of a27r7.

west of a10r7 is east of a62r7.

east of a15r3 is west of a63r3.

northeast of a11r7 is southwest of a33r7.

southwest of a14r3 is northeast of a23r3.

southeast of a17r3 is northwest of a22r3.

southwest of a16r3 is northeast of a35r3.

southwest of a19r6 is northeast of a37r6.

southeast of a21r6 is northwest of a39r6.

northeast of a12r1 is southwest of a41r1.

northeast of a21r6 is southwest of a43r6.

northwest of a16r3 is southeast of a45r3.

southwest of a0r4 is northeast of a49r4.

northwest of a0r4 is southeast of a51r4.

southeast of a2r4 is northwest of a53r4.

east of a6r2 is west of a24r2.

east of a4r0 is west of a8r0.

east of a7r4 is west of a55r4.

northwest of a19r6 is southeast of a47r6.

north of a17r3 is south of a29r3.

north of a14r3 is south of a31r3.

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
the player is in a39r6.

the orientation of the player is 6.
