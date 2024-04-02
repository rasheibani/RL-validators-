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

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "545313.2519132181".
the Y of the a9r0 is "1775351.14099297".

[create a16r0]
a16r0 is a area. "An area (16) in r0".
the printed name of the a16r0 is "Room 0".
Understand "Area 16 in Room 0" as a16r0.
the parent of the a16r0 is r0.
the X of the a16r0 is "546301.9912057454".
the Y of the a16r0 is "1774764.2196130906".

[create a17r0]
a17r0 is a area. "An area (17) in r0".
the printed name of the a17r0 is "Room 0".
Understand "Area 17 in Room 0" as a17r0.
the parent of the a17r0 is r0.
the X of the a17r0 is "546335.5483636888".
the Y of the a17r0 is "1774766.9119265778".

[create a18r0]
a18r0 is a area. "An area (18) in r0".
the printed name of the a18r0 is "Room 0".
Understand "Area 18 in Room 0" as a18r0.
the parent of the a18r0 is r0.
the X of the a18r0 is "546697.0183879724".
the Y of the a18r0 is "1775320.1883249702".

[create a19r0]
a19r0 is a area. "An area (19) in r0".
the printed name of the a19r0 is "Room 0".
Understand "Area 19 in Room 0" as a19r0.
the parent of the a19r0 is r0.
the X of the a19r0 is "546140.6865942029".
the Y of the a19r0 is "1774889.6751110333".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "545222.6098480205".
the Y of the a22r0 is "1775304.665719969".

[create a43r0]
a43r0 is a area. "An area (43) in r0".
the printed name of the a43r0 is "Room 0".
Understand "Area 43 in Room 0" as a43r0.
the parent of the a43r0 is r0.
the X of the a43r0 is "546369.0".
the Y of the a43r0 is "1774653.0".

[create a44r0]
a44r0 is a area. "An area (44) in r0".
the printed name of the a44r0 is "Room 0".
Understand "Area 44 in Room 0" as a44r0.
the parent of the a44r0 is r0.
the X of the a44r0 is "546347.5".
the Y of the a44r0 is "1774738.00390625".

[create a53r0]
a53r0 is a area. "An area (53) in r0".
the printed name of the a53r0 is "Room 0".
Understand "Area 53 in Room 0" as a53r0.
the parent of the a53r0 is r0.
the X of the a53r0 is "545092.0".
the Y of the a53r0 is "1775308.0".

[create a54r0]
a54r0 is a area. "An area (54) in r0".
the printed name of the a54r0 is "Room 0".
Understand "Area 54 in Room 0" as a54r0.
the parent of the a54r0 is r0.
the X of the a54r0 is "545153.9763877088".
the Y of the a54r0 is "1775310.7893767639".

[create a55r0]
a55r0 is a area. "An area (55) in r0".
the printed name of the a55r0 is "Room 0".
Understand "Area 55 in Room 0" as a55r0.
the parent of the a55r0 is r0.
the X of the a55r0 is "545166.0".
the Y of the a55r0 is "1775200.0".

[create a56r0]
a56r0 is a area. "An area (56) in r0".
the printed name of the a56r0 is "Room 0".
Understand "Area 56 in Room 0" as a56r0.
the parent of the a56r0 is r0.
the X of the a56r0 is "545203.9861673415".
the Y of the a56r0 is "1775265.7034850302".

[create a59r0]
a59r0 is a area. "An area (59) in r0".
the printed name of the a59r0 is "Room 0".
Understand "Area 59 in Room 0" as a59r0.
the parent of the a59r0 is r0.
the X of the a59r0 is "546283.0".
the Y of the a59r0 is "1774653.0".

[create a60r0]
a60r0 is a area. "An area (60) in r0".
the printed name of the a60r0 is "Room 0".
Understand "Area 60 in Room 0" as a60r0.
the parent of the a60r0 is r0.
the X of the a60r0 is "546269.0".
the Y of the a60r0 is "1774714.4050772625".

[create a61r0]
a61r0 is a area. "An area (61) in r0".
the printed name of the a61r0 is "Room 0".
Understand "Area 61 in Room 0" as a61r0.
the parent of the a61r0 is r0.
the X of the a61r0 is "546142.0".
the Y of the a61r0 is "1774939.0".

[create a0r1]
a0r1 is a area. "An area (0) in r1".
the printed name of the a0r1 is "Room 1".
Understand "Area 0 in Room 1" as a0r1.
the parent of the a0r1 is r1.
the X of the a0r1 is "546747.7930791576".
the Y of the a0r1 is "1777119.8944909293".

[create a1r1]
a1r1 is a area. "An area (1) in r1".
the printed name of the a1r1 is "Room 1".
Understand "Area 1 in Room 1" as a1r1.
the parent of the a1r1 is r1.
the X of the a1r1 is "547108.6312753353".
the Y of the a1r1 is "1777143.6541939504".

[create a2r1]
a2r1 is a area. "An area (2) in r1".
the printed name of the a2r1 is "Room 1".
Understand "Area 2 in Room 1" as a2r1.
the parent of the a2r1 is r1.
the X of the a2r1 is "547969.7072666293".
the Y of the a2r1 is "1777261.4777749958".

[create a3r1]
a3r1 is a area. "An area (3) in r1".
the printed name of the a3r1 is "Room 1".
Understand "Area 3 in Room 1" as a3r1.
the parent of the a3r1 is r1.
the X of the a3r1 is "548313.3793965105".
the Y of the a3r1 is "1777174.0644405335".

[create a4r1]
a4r1 is a area. "An area (4) in r1".
the printed name of the a4r1 is "Room 1".
Understand "Area 4 in Room 1" as a4r1.
the parent of the a4r1 is r1.
the X of the a4r1 is "548634.3051002845".
the Y of the a4r1 is "1777261.4983535397".

[create a5r1]
a5r1 is a area. "An area (5) in r1".
the printed name of the a5r1 is "Room 1".
Understand "Area 5 in Room 1" as a5r1.
the parent of the a5r1 is r1.
the X of the a5r1 is "545238.9449404762".
the Y of the a5r1 is "1777300.0".

[create a6r1]
a6r1 is a area. "An area (6) in r1".
the printed name of the a6r1 is "Room 1".
Understand "Area 6 in Room 1" as a6r1.
the parent of the a6r1 is r1.
the X of the a6r1 is "545338.4414637961".
the Y of the a6r1 is "1777181.5150352006".

[create a7r1]
a7r1 is a area. "An area (7) in r1".
the printed name of the a7r1 is "Room 1".
Understand "Area 7 in Room 1" as a7r1.
the parent of the a7r1 is r1.
the X of the a7r1 is "547144.668728506".
the Y of the a7r1 is "1777182.463758904".

[create a8r1]
a8r1 is a area. "An area (8) in r1".
the printed name of the a8r1 is "Room 1".
Understand "Area 8 in Room 1" as a8r1.
the parent of the a8r1 is r1.
the X of the a8r1 is "547491.9151692871".
the Y of the a8r1 is "1777261.610847683".

[create a10r1]
a10r1 is a area. "An area (10) in r1".
the printed name of the a10r1 is "Room 1".
Understand "Area 10 in Room 1" as a10r1.
the parent of the a10r1 is r1.
the X of the a10r1 is "546656.4470359627".
the Y of the a10r1 is "1777115.5529640373".

[create a11r1]
a11r1 is a area. "An area (11) in r1".
the printed name of the a11r1 is "Room 1".
Understand "Area 11 in Room 1" as a11r1.
the parent of the a11r1 is r1.
the X of the a11r1 is "547678.1730804583".
the Y of the a11r1 is "1776059.7005024815".

[create a13r1]
a13r1 is a area. "An area (13) in r1".
the printed name of the a13r1 is "Room 1".
Understand "Area 13 in Room 1" as a13r1.
the parent of the a13r1 is r1.
the X of the a13r1 is "545222.6162038349".
the Y of the a13r1 is "1776858.4234448727".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "546418.2535798342".
the Y of the a14r1 is "1777224.3939053".

[create a15r1]
a15r1 is a area. "An area (15) in r1".
the printed name of the a15r1 is "Room 1".
Understand "Area 15 in Room 1" as a15r1.
the parent of the a15r1 is r1.
the X of the a15r1 is "546733.5699375974".
the Y of the a15r1 is "1777103.7257309472".

[create a23r1]
a23r1 is a area. "An area (23) in r1".
the printed name of the a23r1 is "Room 1".
Understand "Area 23 in Room 1" as a23r1.
the parent of the a23r1 is r1.
the X of the a23r1 is "546697.0".
the Y of the a23r1 is "1777314.0".

[create a24r1]
a24r1 is a area. "An area (24) in r1".
the printed name of the a24r1 is "Room 1".
Understand "Area 24 in Room 1" as a24r1.
the parent of the a24r1 is r1.
the X of the a24r1 is "546728.826460481".
the Y of the a24r1 is "1777305.767353952".

[create a25r1]
a25r1 is a area. "An area (25) in r1".
the printed name of the a25r1 is "Room 1".
Understand "Area 25 in Room 1" as a25r1.
the parent of the a25r1 is r1.
the X of the a25r1 is "546378.0".
the Y of the a25r1 is "1777314.0".

[create a26r1]
a26r1 is a area. "An area (26) in r1".
the printed name of the a26r1 is "Room 1".
Understand "Area 26 in Room 1" as a26r1.
the parent of the a26r1 is r1.
the X of the a26r1 is "546409.994017094".
the Y of the a26r1 is "1777305.7505982907".

[create a27r1]
a27r1 is a area. "An area (27) in r1".
the printed name of the a27r1 is "Room 1".
Understand "Area 27 in Room 1" as a27r1.
the parent of the a27r1 is r1.
the X of the a27r1 is "545255.0".
the Y of the a27r1 is "1776788.0".

[create a28r1]
a28r1 is a area. "An area (28) in r1".
the printed name of the a28r1 is "Room 1".
Understand "Area 28 in Room 1" as a28r1.
the parent of the a28r1 is r1.
the X of the a28r1 is "545246.9921842947".
the Y of the a28r1 is "1776825.8338188233".

[create a29r1]
a29r1 is a area. "An area (29) in r1".
the printed name of the a29r1 is "Room 1".
Understand "Area 29 in Room 1" as a29r1.
the parent of the a29r1 is r1.
the X of the a29r1 is "547548.0".
the Y of the a29r1 is "1777224.0".

[create a30r1]
a30r1 is a area. "An area (30) in r1".
the printed name of the a30r1 is "Room 1".
Understand "Area 30 in Room 1" as a30r1.
the parent of the a30r1 is r1.
the X of the a30r1 is "547508.9448424068".
the Y of the a30r1 is "1777242.8194842406".

[create a31r1]
a31r1 is a area. "An area (31) in r1".
the printed name of the a31r1 is "Room 1".
Understand "Area 31 in Room 1" as a31r1.
the parent of the a31r1 is r1.
the X of the a31r1 is "547913.0".
the Y of the a31r1 is "1777224.0".

[create a32r1]
a32r1 is a area. "An area (32) in r1".
the printed name of the a32r1 is "Room 1".
Understand "Area 32 in Room 1" as a32r1.
the parent of the a32r1 is r1.
the X of the a32r1 is "547951.7029286583".
the Y of the a32r1 is "1777242.4774523713".

[create a33r1]
a33r1 is a area. "An area (33) in r1".
the printed name of the a33r1 is "Room 1".
Understand "Area 33 in Room 1" as a33r1.
the parent of the a33r1 is r1.
the X of the a33r1 is "548691.0".
the Y of the a33r1 is "1777224.0".

[create a34r1]
a34r1 is a area. "An area (34) in r1".
the printed name of the a34r1 is "Room 1".
Understand "Area 34 in Room 1" as a34r1.
the parent of the a34r1 is r1.
the X of the a34r1 is "548651.9982014389".
the Y of the a34r1 is "1777242.4442446043".

[create a35r1]
a35r1 is a area. "An area (35) in r1".
the printed name of the a35r1 is "Room 1".
Understand "Area 35 in Room 1" as a35r1.
the parent of the a35r1 is r1.
the X of the a35r1 is "548681.0".
the Y of the a35r1 is "1777314.0".

[create a36r1]
a36r1 is a area. "An area (36) in r1".
the printed name of the a36r1 is "Room 1".
Understand "Area 36 in Room 1" as a36r1.
the parent of the a36r1 is r1.
the X of the a36r1 is "548650.1050372067".
the Y of the a36r1 is "1777287.7894485786".

[create a37r1]
a37r1 is a area. "An area (37) in r1".
the printed name of the a37r1 is "Room 1".
Understand "Area 37 in Room 1" as a37r1.
the parent of the a37r1 is r1.
the X of the a37r1 is "547923.0".
the Y of the a37r1 is "1777314.0".

[create a38r1]
a38r1 is a area. "An area (38) in r1".
the printed name of the a38r1 is "Room 1".
Understand "Area 38 in Room 1" as a38r1.
the parent of the a38r1 is r1.
the X of the a38r1 is "547953.8949627937".
the Y of the a38r1 is "1777287.7894485786".

[create a39r1]
a39r1 is a area. "An area (39) in r1".
the printed name of the a39r1 is "Room 1".
Understand "Area 39 in Room 1" as a39r1.
the parent of the a39r1 is r1.
the X of the a39r1 is "547539.0".
the Y of the a39r1 is "1777314.0".

[create a40r1]
a40r1 is a area. "An area (40) in r1".
the printed name of the a40r1 is "Room 1".
Understand "Area 40 in Room 1" as a40r1.
the parent of the a40r1 is r1.
the X of the a40r1 is "547507.9123931624".
the Y of the a40r1 is "1777288.166239316".

[create a41r1]
a41r1 is a area. "An area (41) in r1".
the printed name of the a41r1 is "Room 1".
Understand "Area 41 in Room 1" as a41r1.
the parent of the a41r1 is r1.
the X of the a41r1 is "547189.0".
the Y of the a41r1 is "1777314.0".

[create a42r1]
a42r1 is a area. "An area (42) in r1".
the printed name of the a42r1 is "Room 1".
Understand "Area 42 in Room 1" as a42r1.
the parent of the a42r1 is r1.
the X of the a42r1 is "547157.1923615086".
the Y of the a42r1 is "1777305.4040895505".

[create a57r1]
a57r1 is a area. "An area (57) in r1".
the printed name of the a57r1 is "Room 1".
Understand "Area 57 in Room 1" as a57r1.
the parent of the a57r1 is r1.
the X of the a57r1 is "545172.0".
the Y of the a57r1 is "1776825.0".

[create a58r1]
a58r1 is a area. "An area (58) in r1".
the printed name of the a58r1 is "Room 1".
Understand "Area 58 in Room 1" as a58r1.
the parent of the a58r1 is r1.
the X of the a58r1 is "545209.8974585026".
the Y of the a58r1 is "1776854.2159204248".

[create a62r1]
a62r1 is a area. "An area (62) in r1".
the printed name of the a62r1 is "Room 1".
Understand "Area 62 in Room 1" as a62r1.
the parent of the a62r1 is r1.
the X of the a62r1 is "545209.0".
the Y of the a62r1 is "1777277.0".

[create a12r2]
a12r2 is a area. "An area (12) in r2".
the printed name of the a12r2 is "Room 2".
Understand "Area 12 in Room 2" as a12r2.
the parent of the a12r2 is r2.
the X of the a12r2 is "547683.0000000001".
the Y of the a12r2 is "1775368.577830189".

[create a20r2]
a20r2 is a area. "An area (20) in r2".
the printed name of the a20r2 is "Room 2".
Understand "Area 20 in Room 2" as a20r2.
the parent of the a20r2 is r2.
the X of the a20r2 is "548049.7290448342".
the Y of the a20r2 is "1775245.062703054".

[create a21r2]
a21r2 is a area. "An area (21) in r2".
the printed name of the a21r2 is "Room 2".
Understand "Area 21 in Room 2" as a21r2.
the parent of the a21r2 is r2.
the X of the a21r2 is "547301.3945333334".
the Y of the a21r2 is "1775240.268666667".

[create a45r2]
a45r2 is a area. "An area (45) in r2".
the printed name of the a45r2 is "Room 2".
Understand "Area 45 in Room 2" as a45r2.
the parent of the a45r2 is r2.
the X of the a45r2 is "547256.0".
the Y of the a45r2 is "1775191.0".

[create a46r2]
a46r2 is a area. "An area (46) in r2".
the printed name of the a46r2 is "Room 2".
Understand "Area 46 in Room 2" as a46r2.
the parent of the a46r2 is r2.
the X of the a46r2 is "547287.2137161084".
the Y of the a46r2 is "1775216.633971292".

[create a47r2]
a47r2 is a area. "An area (47) in r2".
the printed name of the a47r2 is "Room 2".
Understand "Area 47 in Room 2" as a47r2.
the parent of the a47r2 is r2.
the X of the a47r2 is "547247.0".
the Y of the a47r2 is "1775280.0".

[create a48r2]
a48r2 is a area. "An area (48) in r2".
the printed name of the a48r2 is "Room 2".
Understand "Area 48 in Room 2" as a48r2.
the parent of the a48r2 is r2.
the X of the a48r2 is "547277.8730024246".
the Y of the a48r2 is "1775260.6444609195".

[create a49r2]
a49r2 is a area. "An area (49) in r2".
the printed name of the a49r2 is "Room 2".
Understand "Area 49 in Room 2" as a49r2.
the parent of the a49r2 is r2.
the X of the a49r2 is "548098.0".
the Y of the a49r2 is "1775191.0".

[create a50r2]
a50r2 is a area. "An area (50) in r2".
the printed name of the a50r2 is "Room 2".
Understand "Area 50 in Room 2" as a50r2.
the parent of the a50r2 is r2.
the X of the a50r2 is "548066.7862838916".
the Y of the a50r2 is "1775216.633971292".

[create a51r2]
a51r2 is a area. "An area (51) in r2".
the printed name of the a51r2 is "Room 2".
Understand "Area 51 in Room 2" as a51r2.
the parent of the a51r2 is r2.
the X of the a51r2 is "548107.0".
the Y of the a51r2 is "1775280.0".

[create a52r2]
a52r2 is a area. "An area (52) in r2".
the printed name of the a52r2 is "Room 2".
Understand "Area 52 in Room 2" as a52r2.
the parent of the a52r2 is r2.
the X of the a52r2 is "548063.9383292026".
the Y of the a52r2 is "1775261.87702289".

[create d0]
d0 is a door. "A door between a9r0 and a10r1".
d0 is northeast of a9r0 and southwest of a10r1.

[create d1]
d1 is a door. "A door between a18r0 and a15r1".
d1 is north of a18r0 and south of a15r1.

[create d4]
d4 is a door. "A door between a11r1 and a12r2".
d4 is south of a11r1 and north of a12r2.

east of a0r1 is west of a1r1.

east of a2r1 is west of a3r1.

east of a3r1 is west of a4r1.

southeast of a5r1 is northwest of a6r1.

east of a7r1 is west of a8r1.

southwest of a3r1 is northeast of a11r1.

southeast of a1r1 is northwest of a11r1.

south of a6r1 is north of a13r1.

east of a6r1 is west of a14r1.

east of a10r1 is west of a15r1.

northwest of a10r1 is southeast of a14r1.

east of a16r0 is west of a17r0.

northeast of a17r0 is southwest of a18r0.

northwest of a16r0 is southeast of a19r0.

east of a12r2 is west of a20r2.

west of a12r2 is east of a21r2.

east of a9r0 is west of a18r0.

southwest of a9r0 is northeast of a22r0.

east of a23r1 is west of a24r1.

east of a25r1 is west of a26r1.

north of a27r1 is south of a28r1.

northwest of a29r1 is southeast of a30r1.

northeast of a31r1 is southwest of a32r1.

northwest of a33r1 is southeast of a34r1.

southwest of a35r1 is northeast of a36r1.

southeast of a37r1 is northwest of a38r1.

southwest of a39r1 is northeast of a40r1.

west of a41r1 is east of a42r1.

north of a43r0 is south of a44r0.

northeast of a45r2 is southwest of a46r2.

southeast of a47r2 is northwest of a48r2.

northwest of a49r2 is southeast of a50r2.

southwest of a51r2 is northeast of a52r2.

east of a53r0 is west of a54r0.

northeast of a55r0 is southwest of a56r0.

northeast of a57r1 is southwest of a58r1.

north of a59r0 is south of a60r0.

north of a19r0 is south of a61r0.

southwest of a5r1 is northeast of a62r1.

southeast of a4r1 is northwest of a34r1.

northeast of a4r1 is southwest of a36r1.

northwest of a2r1 is southeast of a38r1.

northeast of a8r1 is southwest of a40r1.

north of a7r1 is south of a42r1.

north of a0r1 is south of a24r1.

north of a14r1 is south of a26r1.

west of a22r0 is east of a54r0.

southwest of a22r0 is northeast of a56r0.

west of a13r1 is east of a58r1.

southeast of a13r1 is northwest of a28r1.

southwest of a0r1 is northeast of a15r1.

northeast of a1r1 is southwest of a7r1.

southeast of a8r1 is northwest of a30r1.

southwest of a2r1 is northeast of a32r1.

southwest of a16r0 is northeast of a60r0.

south of a17r0 is north of a44r0.

southwest of a21r2 is northeast of a46r2.

northwest of a21r2 is southeast of a48r2.

southeast of a20r2 is northwest of a50r2.

northeast of a20r2 is southwest of a52r2.

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
the player is in a22r0.

the orientation of the player is 6.
