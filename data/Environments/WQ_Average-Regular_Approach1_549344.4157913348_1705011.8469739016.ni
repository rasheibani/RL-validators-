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

[create a5r0]
a5r0 is a area. "An area (5) in r0".
the printed name of the a5r0 is "Room 0".
Understand "Area 5 in Room 0" as a5r0.
the parent of the a5r0 is r0.
the X of the a5r0 is "547342.4921500895".
the Y of the a5r0 is "1705060.7986347184".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "547676.9486065779".
the Y of the a6r0 is "1704978.1064398694".

[create a7r0]
a7r0 is a area. "An area (7) in r0".
the printed name of the a7r0 is "Room 0".
Understand "Area 7 in Room 0" as a7r0.
the parent of the a7r0 is r0.
the X of the a7r0 is "548006.5509290383".
the Y of the a7r0 is "1705061.0062088708".

[create a10r0]
a10r0 is a area. "An area (10) in r0".
the printed name of the a10r0 is "Room 0".
Understand "Area 10 in Room 0" as a10r0.
the parent of the a10r0 is r0.
the X of the a10r0 is "547781.0301712437".
the Y of the a10r0 is "1704114.7782258065".

[create a11r0]
a11r0 is a area. "An area (11) in r0".
the printed name of the a11r0 is "Room 0".
Understand "Area 11 in Room 0" as a11r0.
the parent of the a11r0 is r0.
the X of the a11r0 is "547751.7472466428".
the Y of the a11r0 is "1704355.4311910914".

[create a12r0]
a12r0 is a area. "An area (12) in r0".
the printed name of the a12r0 is "Room 0".
Understand "Area 12 in Room 0" as a12r0.
the parent of the a12r0 is r0.
the X of the a12r0 is "547495.687052676".
the Y of the a12r0 is "1704119.907804485".

[create a13r0]
a13r0 is a area. "An area (13) in r0".
the printed name of the a13r0 is "Room 0".
Understand "Area 13 in Room 0" as a13r0.
the parent of the a13r0 is r0.
the X of the a13r0 is "547512.9425430542".
the Y of the a13r0 is "1704192.6730880323".

[create a14r0]
a14r0 is a area. "An area (14) in r0".
the printed name of the a14r0 is "Room 0".
Understand "Area 14 in Room 0" as a14r0.
the parent of the a14r0 is r0.
the X of the a14r0 is "547637.091129236".
the Y of the a14r0 is "1704383.8659082085".

[create a15r0]
a15r0 is a area. "An area (15) in r0".
the printed name of the a15r0 is "Room 0".
Understand "Area 15 in Room 0" as a15r0.
the parent of the a15r0 is r0.
the X of the a15r0 is "549344.4157913348".
the Y of the a15r0 is "1705011.8469739016".

[create a32r0]
a32r0 is a area. "An area (32) in r0".
the printed name of the a32r0 is "Room 0".
Understand "Area 32 in Room 0" as a32r0.
the parent of the a32r0 is r0.
the X of the a32r0 is "547286.0".
the Y of the a32r0 is "1705024.0".

[create a33r0]
a33r0 is a area. "An area (33) in r0".
the printed name of the a33r0 is "Room 0".
Understand "Area 33 in Room 0" as a33r0.
the parent of the a33r0 is r0.
the X of the a33r0 is "547326.7622699386".
the Y of the a33r0 is "1705042.6487730062".

[create a34r0]
a34r0 is a area. "An area (34) in r0".
the printed name of the a34r0 is "Room 0".
Understand "Area 34 in Room 0" as a34r0.
the parent of the a34r0 is r0.
the X of the a34r0 is "548063.0".
the Y of the a34r0 is "1705024.0".

[create a35r0]
a35r0 is a area. "An area (35) in r0".
the printed name of the a35r0 is "Room 0".
Understand "Area 35 in Room 0" as a35r0.
the parent of the a35r0 is r0.
the X of the a35r0 is "548022.6625777469".
the Y of the a35r0 is "1705042.6912577746".

[create a44r0]
a44r0 is a area. "An area (44) in r0".
the printed name of the a44r0 is "Room 0".
Understand "Area 44 in Room 0" as a44r0.
the parent of the a44r0 is r0.
the X of the a44r0 is "547295.0".
the Y of the a44r0 is "1705114.0".

[create a45r0]
a45r0 is a area. "An area (45) in r0".
the printed name of the a45r0 is "Room 0".
Understand "Area 45 in Room 0" as a45r0.
the parent of the a45r0 is r0.
the X of the a45r0 is "547325.915378007".
the Y of the a45r0 is "1705088.1834621993".

[create a46r0]
a46r0 is a area. "An area (46) in r0".
the printed name of the a46r0 is "Room 0".
Understand "Area 46 in Room 0" as a46r0.
the parent of the a46r0 is r0.
the X of the a46r0 is "548054.0".
the Y of the a46r0 is "1705114.0".

[create a47r0]
a47r0 is a area. "An area (47) in r0".
the printed name of the a47r0 is "Room 0".
Understand "Area 47 in Room 0" as a47r0.
the parent of the a47r0 is r0.
the X of the a47r0 is "548022.9123931624".
the Y of the a47r0 is "1705088.166239316".

[create a63r0]
a63r0 is a area. "An area (63) in r0".
the printed name of the a63r0 is "Room 0".
Understand "Area 63 in Room 0" as a63r0.
the parent of the a63r0 is r0.
the X of the a63r0 is "549377.8444444444".
the Y of the a63r0 is "1704997.0".

[create a20r1]
a20r1 is a area. "An area (20) in r1".
the printed name of the a20r1 is "Room 1".
Understand "Area 20 in Room 1" as a20r1.
the parent of the a20r1 is r1.
the X of the a20r1 is "547166.2896516463".
the Y of the a20r1 is "1703160.883319583".

[create a21r1]
a21r1 is a area. "An area (21) in r1".
the printed name of the a21r1 is "Room 1".
Understand "Area 21 in Room 1" as a21r1.
the parent of the a21r1 is r1.
the X of the a21r1 is "547260.4539705468".
the Y of the a21r1 is "1703048.7643334025".

[create a23r1]
a23r1 is a area. "An area (23) in r1".
the printed name of the a23r1 is "Room 1".
Understand "Area 23 in Room 1" as a23r1.
the parent of the a23r1 is r1.
the X of the a23r1 is "547144.2390740269".
the Y of the a23r1 is "1703162.4312827692".

[create a31r1]
a31r1 is a area. "An area (31) in r1".
the printed name of the a31r1 is "Room 1".
Understand "Area 31 in Room 1" as a31r1.
the parent of the a31r1 is r1.
the X of the a31r1 is "547048.7043713275".
the Y of the a31r1 is "1703022.7067112252".

[create a62r1]
a62r1 is a area. "An area (62) in r1".
the printed name of the a62r1 is "Room 1".
Understand "Area 62 in Room 1" as a62r1.
the parent of the a62r1 is r1.
the X of the a62r1 is "547301.0".
the Y of the a62r1 is "1703000.0".

[create a68r1]
a68r1 is a area. "An area (68) in r1".
the printed name of the a68r1 is "Room 1".
Understand "Area 68 in Room 1" as a68r1.
the parent of the a68r1 is r1.
the X of the a68r1 is "547018.0".
the Y of the a68r1 is "1702972.0".

[create a0r2]
a0r2 is a area. "An area (0) in r2".
the printed name of the a0r2 is "Room 2".
Understand "Area 0 in Room 2" as a0r2.
the parent of the a0r2 is r2.
the X of the a0r2 is "546477.3333333333".
the Y of the a0r2 is "1705060.9694444444".

[create a1r2]
a1r2 is a area. "An area (1) in r2".
the printed name of the a1r2 is "Room 2".
Understand "Area 1 in Room 2" as a1r2.
the parent of the a1r2 is r2.
the X of the a1r2 is "546544.5167685037".
the Y of the a1r2 is "1704973.513395759".

[create a17r2]
a17r2 is a area. "An area (17) in r2".
the printed name of the a17r2 is "Room 2".
Understand "Area 17 in Room 2" as a17r2.
the parent of the a17r2 is r2.
the X of the a17r2 is "546549.8708448356".
the Y of the a17r2 is "1704805.3162913683".

[create a40r2]
a40r2 is a area. "An area (40) in r2".
the printed name of the a40r2 is "Room 2".
Understand "Area 40 in Room 2" as a40r2.
the parent of the a40r2 is r2.
the X of the a40r2 is "546644.0".
the Y of the a40r2 is "1705104.0".

[create a41r2]
a41r2 is a area. "An area (41) in r2".
the printed name of the a41r2 is "Room 2".
Understand "Area 41 in Room 2" as a41r2.
the parent of the a41r2 is r2.
the X of the a41r2 is "546610.666666667".
the Y of the a41r2 is "1705053.5364495006".

[create a65r2]
a65r2 is a area. "An area (65) in r2".
the printed name of the a65r2 is "Room 2".
Understand "Area 65 in Room 2" as a65r2.
the parent of the a65r2 is r2.
the X of the a65r2 is "546444.0".
the Y of the a65r2 is "1705104.0".

[create a18r3]
a18r3 is a area. "An area (18) in r3".
the printed name of the a18r3 is "Room 3".
Understand "Area 18 in Room 3" as a18r3.
the parent of the a18r3 is r3.
the X of the a18r3 is "545903.7496886395".
the Y of the a18r3 is "1703162.317373606".

[create a19r3]
a19r3 is a area. "An area (19) in r3".
the printed name of the a19r3 is "Room 3".
Understand "Area 19 in Room 3" as a19r3.
the parent of the a19r3 is r3.
the X of the a19r3 is "545997.2968536066".
the Y of the a19r3 is "1703051.42097439".

[create a22r3]
a22r3 is a area. "An area (22) in r3".
the printed name of the a22r3 is "Room 3".
Understand "Area 22 in Room 3" as a22r3.
the parent of the a22r3 is r3.
the X of the a22r3 is "545890.0086022713".
the Y of the a22r3 is "1703164.0104014394".

[create a30r3]
a30r3 is a area. "An area (30) in r3".
the printed name of the a30r3 is "Room 3".
Understand "Area 30 in Room 3" as a30r3.
the parent of the a30r3 is r3.
the X of the a30r3 is "545786.8816727516".
the Y of the a30r3 is "1703019.495295859".

[create a67r3]
a67r3 is a area. "An area (67) in r3".
the printed name of the a67r3 is "Room 3".
Understand "Area 67 in Room 3" as a67r3.
the parent of the a67r3 is r3.
the X of the a67r3 is "545756.0".
the Y of the a67r3 is "1702972.0".

[create a69r3]
a69r3 is a area. "An area (69) in r3".
the printed name of the a69r3 is "Room 3".
Understand "Area 69 in Room 3" as a69r3.
the parent of the a69r3 is r3.
the X of the a69r3 is "546038.0".
the Y of the a69r3 is "1703000.0".

[create a2r4]
a2r4 is a area. "An area (2) in r4".
the printed name of the a2r4 is "Room 4".
Understand "Area 2 in Room 4" as a2r4.
the parent of the a2r4 is r4.
the X of the a2r4 is "545050.2813718207".
the Y of the a2r4 is "1705061.2845894434".

[create a3r4]
a3r4 is a area. "An area (3) in r4".
the printed name of the a3r4 is "Room 4".
Understand "Area 3 in Room 4" as a3r4.
the parent of the a3r4 is r4.
the X of the a3r4 is "545405.7977073499".
the Y of the a3r4 is "1704944.5677680378".

[create a4r4]
a4r4 is a area. "An area (4) in r4".
the printed name of the a4r4 is "Room 4".
Understand "Area 4 in Room 4" as a4r4.
the parent of the a4r4 is r4.
the X of the a4r4 is "545779.1653756485".
the Y of the a4r4 is "1705061.2658517454".

[create a36r4]
a36r4 is a area. "An area (36) in r4".
the printed name of the a36r4 is "Room 4".
Understand "Area 36 in Room 4" as a36r4.
the parent of the a36r4 is r4.
the X of the a36r4 is "544994.0".
the Y of the a36r4 is "1705024.0".

[create a37r4]
a37r4 is a area. "An area (37) in r4".
the printed name of the a37r4 is "Room 4".
Understand "Area 37 in Room 4" as a37r4.
the parent of the a37r4 is r4.
the X of the a37r4 is "545033.4941364606".
the Y of the a37r4 is "1705042.7755863538".

[create a38r4]
a38r4 is a area. "An area (38) in r4".
the printed name of the a38r4 is "Room 4".
Understand "Area 38 in Room 4" as a38r4.
the parent of the a38r4 is r4.
the X of the a38r4 is "545836.0".
the Y of the a38r4 is "1705024.0".

[create a39r4]
a39r4 is a area. "An area (39) in r4".
the printed name of the a39r4 is "Room 4".
Understand "Area 39 in Room 4" as a39r4.
the parent of the a39r4 is r4.
the X of the a39r4 is "545795.6686930091".
the Y of the a39r4 is "1705042.2965214453".

[create a42r4]
a42r4 is a area. "An area (42) in r4".
the printed name of the a42r4 is "Room 4".
Understand "Area 42 in Room 4" as a42r4.
the parent of the a42r4 is r4.
the X of the a42r4 is "545826.0".
the Y of the a42r4 is "1705114.0".

[create a43r4]
a43r4 is a area. "An area (43) in r4".
the printed name of the a43r4 is "Room 4".
Understand "Area 43 in Room 4" as a43r4.
the parent of the a43r4 is r4.
the X of the a43r4 is "545795.1050372066".
the Y of the a43r4 is "1705087.7894485788".

[create a48r4]
a48r4 is a area. "An area (48) in r4".
the printed name of the a48r4 is "Room 4".
Understand "Area 48 in Room 4" as a48r4.
the parent of the a48r4 is r4.
the X of the a48r4 is "545003.0".
the Y of the a48r4 is "1705114.0".

[create a49r4]
a49r4 is a area. "An area (49) in r4".
the printed name of the a49r4 is "Room 4".
Understand "Area 49 in Room 4" as a49r4.
the parent of the a49r4 is r4.
the X of the a49r4 is "545034.087606838".
the Y of the a49r4 is "1705088.166239316".

[create a24r5]
a24r5 is a area. "An area (24) in r5".
the printed name of the a24r5 is "Room 5".
Understand "Area 24 in Room 5" as a24r5.
the parent of the a24r5 is r5.
the X of the a24r5 is "549265.0253397928".
the Y of the a24r5 is "1703217.9152977373".

[create a25r5]
a25r5 is a area. "An area (25) in r5".
the printed name of the a25r5 is "Room 5".
Understand "Area 25 in Room 5" as a25r5.
the parent of the a25r5 is r5.
the X of the a25r5 is "547812.5884030655".
the Y of the a25r5 is "1703568.8018806516".

[create a26r5]
a26r5 is a area. "An area (26) in r5".
the printed name of the a26r5 is "Room 5".
Understand "Area 26 in Room 5" as a26r5.
the parent of the a26r5 is r5.
the X of the a26r5 is "549684.795286621".
the Y of the a26r5 is "1702490.5977945551".

[create a27r5]
a27r5 is a area. "An area (27) in r5".
the printed name of the a27r5 is "Room 5".
Understand "Area 27 in Room 5" as a27r5.
the parent of the a27r5 is r5.
the X of the a27r5 is "549872.0661764705".
the Y of the a27r5 is "1702487.0".

[create a28r5]
a28r5 is a area. "An area (28) in r5".
the printed name of the a28r5 is "Room 5".
Understand "Area 28 in Room 5" as a28r5.
the parent of the a28r5 is r5.
the X of the a28r5 is "548850.8480141502".
the Y of the a28r5 is "1702983.1248914616".

[create a29r5]
a29r5 is a area. "An area (29) in r5".
the printed name of the a29r5 is "Room 5".
Understand "Area 29 in Room 5" as a29r5.
the parent of the a29r5 is r5.
the X of the a29r5 is "548172.2997365495".
the Y of the a29r5 is "1703095.2244149852".

[create a50r5]
a50r5 is a area. "An area (50) in r5".
the printed name of the a50r5 is "Room 5".
Understand "Area 50 in Room 5" as a50r5.
the parent of the a50r5 is r5.
the X of the a50r5 is "549672.0".
the Y of the a50r5 is "1702333.0".

[create a51r5]
a51r5 is a area. "An area (51) in r5".
the printed name of the a51r5 is "Room 5".
Understand "Area 51 in Room 5" as a51r5.
the parent of the a51r5 is r5.
the X of the a51r5 is "549676.2013157896".
the Y of the a51r5 is "1702406.8065789475".

[create a52r5]
a52r5 is a area. "An area (52) in r5".
the printed name of the a52r5 is "Room 5".
Understand "Area 52 in Room 5" as a52r5.
the parent of the a52r5 is r5.
the X of the a52r5 is "548108.0".
the Y of the a52r5 is "1702951.0".

[create a53r5]
a53r5 is a area. "An area (53) in r5".
the printed name of the a53r5 is "Room 5".
Understand "Area 53 in Room 5" as a53r5.
the parent of the a53r5 is r5.
the X of the a53r5 is "548144.0".
the Y of the a53r5 is "1703032.3823529412".

[create a54r5]
a54r5 is a area. "An area (54) in r5".
the printed name of the a54r5 is "Room 5".
Understand "Area 54 in Room 5" as a54r5.
the parent of the a54r5 is r5.
the X of the a54r5 is "549352.0".
the Y of the a54r5 is "1703101.0".

[create a55r5]
a55r5 is a area. "An area (55) in r5".
the printed name of the a55r5 is "Room 5".
Understand "Area 55 in Room 5" as a55r5.
the parent of the a55r5 is r5.
the X of the a55r5 is "549296.9915114488".
the Y of the a55r5 is "1703177.2877525997".

[create a56r5]
a56r5 is a area. "An area (56) in r5".
the printed name of the a56r5 is "Room 5".
Understand "Area 56 in Room 5" as a56r5.
the parent of the a56r5 is r5.
the X of the a56r5 is "547620.0".
the Y of the a56r5 is "1703495.0".

[create a57r5]
a57r5 is a area. "An area (57) in r5".
the printed name of the a57r5 is "Room 5".
Understand "Area 57 in Room 5" as a57r5.
the parent of the a57r5 is r5.
the X of the a57r5 is "547718.3825136607".
the Y of the a57r5 is "1703534.1666666665".

[create a58r5]
a58r5 is a area. "An area (58) in r5".
the printed name of the a58r5 is "Room 5".
Understand "Area 58 in Room 5" as a58r5.
the parent of the a58r5 is r5.
the X of the a58r5 is "549733.0".
the Y of the a58r5 is "1703731.0".

[create a66r5]
a66r5 is a area. "An area (66) in r5".
the printed name of the a66r5 is "Room 5".
Understand "Area 66 in Room 5" as a66r5.
the parent of the a66r5 is r5.
the X of the a66r5 is "549927.0".
the Y of the a66r5 is "1702521.0".

[create a8r6]
a8r6 is a area. "An area (8) in r6".
the printed name of the a8r6 is "Room 6".
Understand "Area 8 in Room 6" as a8r6.
the parent of the a8r6 is r6.
the X of the a8r6 is "549568.3476716748".
the Y of the a8r6 is "1703775.701377194".

[create a9r6]
a9r6 is a area. "An area (9) in r6".
the printed name of the a9r6 is "Room 6".
Understand "Area 9 in Room 6" as a9r6.
the parent of the a9r6 is r6.
the X of the a9r6 is "549570.5359105705".
the Y of the a9r6 is "1704540.643192971".

[create a16r6]
a16r6 is a area. "An area (16) in r6".
the printed name of the a16r6 is "Room 6".
Understand "Area 16 in Room 6" as a16r6.
the parent of the a16r6 is r6.
the X of the a16r6 is "549424.3602366083".
the Y of the a16r6 is "1704925.6012755313".

[create a59r6]
a59r6 is a area. "An area (59) in r6".
the printed name of the a59r6 is "Room 6".
Understand "Area 59 in Room 6" as a59r6.
the parent of the a59r6 is r6.
the X of the a59r6 is "549606.3318897632".
the Y of the a59r6 is "1703764.5833333335".

[create a60r6]
a60r6 is a area. "An area (60) in r6".
the printed name of the a60r6 is "Room 6".
Understand "Area 60 in Room 6" as a60r6.
the parent of the a60r6 is r6.
the X of the a60r6 is "549733.0".
the Y of the a60r6 is "1704604.0".

[create a61r6]
a61r6 is a area. "An area (61) in r6".
the printed name of the a61r6 is "Room 6".
Understand "Area 61 in Room 6" as a61r6.
the parent of the a61r6 is r6.
the X of the a61r6 is "549638.4970438061".
the Y of the a61r6 is "1704564.8333333335".

[create a64r6]
a64r6 is a area. "An area (64) in r6".
the printed name of the a64r6 is "Room 6".
Understand "Area 64 in Room 6" as a64r6.
the parent of the a64r6 is r6.
the X of the a64r6 is "549411.0".
the Y of the a64r6 is "1704960.31147541".

[create a63r7]
a63r7 is a area. "An area (63) in r7".
the printed name of the a63r7 is "Room 7".
Understand "Area 63 in Room 7" as a63r7.
the parent of the a63r7 is r7.
the X of the a63r7 is "549377.8444444444".
the Y of the a63r7 is "1704997.0".

[create a64r7]
a64r7 is a area. "An area (64) in r7".
the printed name of the a64r7 is "Room 7".
Understand "Area 64 in Room 7" as a64r7.
the parent of the a64r7 is r7.
the X of the a64r7 is "549411.0".
the Y of the a64r7 is "1704960.31147541".

[create d0]
d0 is a door. "A door between a12r0 and a20r1".
d0 is south of a12r0 and north of a20r1.

[create d2]
d2 is a door. "A door between a17r2 and a18r3".
d2 is south of a17r2 and north of a18r3.

[create d4]
d4 is a door. "A door between a22r3 and a3r4".
d4 is north of a22r3 and south of a3r4.

[create d6]
d6 is a door. "A door between a23r1 and a17r2".
d6 is northwest of a23r1 and southeast of a17r2.

[create d8]
d8 is a door. "A door between a24r5 and a8r6".
d8 is northeast of a24r5 and southwest of a8r6.

[create d9]
d9 is a door. "A door between a58r5 and a59r6".
d9 is west of a58r5 and east of a59r6.

[create d10]
d10 is a door. "A door between a10r0 and a25r5".
d10 is south of a10r0 and north of a25r5.

southeast of a0r2 is northwest of a1r2.

east of a2r4 is west of a3r4.

east of a3r4 is west of a4r4.

east of a5r0 is west of a6r0.

east of a6r0 is west of a7r0.

north of a8r6 is south of a9r6.

north of a10r0 is south of a11r0.

north of a12r0 is south of a13r0.

northeast of a13r0 is southwest of a14r0.

south of a6r0 is north of a14r0.

east of a11r0 is west of a15r0.

north of a9r6 is south of a16r6.

south of a1r2 is north of a17r2.

southeast of a18r3 is northwest of a19r3.

southeast of a20r1 is northwest of a21r1.

east of a26r5 is west of a27r5.

northwest of a26r5 is southeast of a28r5.

west of a28r5 is east of a29r5.

southwest of a24r5 is northeast of a28r5.

southwest of a22r3 is northeast of a30r3.

southwest of a23r1 is northeast of a31r1.

southeast of a25r5 is northwest of a29r5.

northeast of a32r0 is southwest of a33r0.

northwest of a34r0 is southeast of a35r0.

northeast of a36r4 is southwest of a37r4.

northwest of a38r4 is southeast of a39r4.

southwest of a40r2 is northeast of a41r2.

southwest of a42r4 is northeast of a43r4.

southeast of a44r0 is northwest of a45r0.

southwest of a46r0 is northeast of a47r0.

southeast of a48r4 is northwest of a49r4.

north of a50r5 is south of a51r5.

northeast of a52r5 is southwest of a53r5.

northwest of a54r5 is southeast of a55r5.

east of a56r5 is west of a57r5.

southwest of a60r6 is northeast of a61r6.

southeast of a21r1 is northwest of a62r1.

southeast of a15r0 is northwest of a63r0.

north of a16r6 is south of a64r6.

northwest of a0r2 is southeast of a65r2.

northeast of a27r5 is southwest of a66r5.

southwest of a30r3 is northeast of a67r3.

southwest of a31r1 is northeast of a68r1.

southeast of a19r3 is northwest of a69r3.

south of a26r5 is north of a51r5.

southwest of a29r5 is northeast of a53r5.

west of a20r1 is east of a23r1.

west of a18r3 is east of a22r3.

southeast of a24r5 is northwest of a55r5.

west of a25r5 is east of a57r5.

east of a8r6 is west of a59r6.

west of a11r0 is east of a14r0.

northwest of a5r0 is southeast of a45r0.

northeast of a7r0 is southwest of a47r0.

northwest of a2r4 is southeast of a49r4.

east of a9r6 is west of a61r6.

southwest of a5r0 is northeast of a33r0.

southeast of a7r0 is northwest of a35r0.

southwest of a2r4 is northeast of a37r4.

southeast of a4r4 is northwest of a39r4.

northeast of a1r2 is southwest of a41r2.

northeast of a4r4 is southwest of a43r4.

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
the player is in a15r0.

the orientation of the player is 6.
