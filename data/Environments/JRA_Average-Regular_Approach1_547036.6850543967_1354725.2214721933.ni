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

[create a12r0]
a12r0 is a area. "An area (12) in r0".
the printed name of the a12r0 is "Room 0".
Understand "Area 12 in Room 0" as a12r0.
the parent of the a12r0 is r0.
the X of the a12r0 is "546461.0".
the Y of the a12r0 is "1353391.0".

[create a13r0]
a13r0 is a area. "An area (13) in r0".
the printed name of the a13r0 is "Room 0".
Understand "Area 13 in Room 0" as a13r0.
the parent of the a13r0 is r0.
the X of the a13r0 is "546428.9473578264".
the Y of the a13r0 is "1353417.1323193454".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "546471.0".
the Y of the a22r0 is "1353480.0".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "546427.3908390412".
the Y of the a23r0 is "1353462.3690068494".

[create a28r0]
a28r0 is a area. "An area (28) in r0".
the printed name of the a28r0 is "Room 0".
Understand "Area 28 in Room 0" as a28r0.
the parent of the a28r0 is r0.
the X of the a28r0 is "545089.0".
the Y of the a28r0 is "1352853.0".

[create a29r0]
a29r0 is a area. "An area (29) in r0".
the printed name of the a29r0 is "Room 0".
Understand "Area 29 in Room 0" as a29r0.
the parent of the a29r0 is r0.
the X of the a29r0 is "545075.0".
the Y of the a29r0 is "1352914.4050772628".

[create a30r0]
a30r0 is a area. "An area (30) in r0".
the printed name of the a30r0 is "Room 0".
Understand "Area 30 in Room 0" as a30r0.
the parent of the a30r0 is r0.
the X of the a30r0 is "545175.0".
the Y of the a30r0 is "1352853.0".

[create a31r0]
a31r0 is a area. "An area (31) in r0".
the printed name of the a31r0 is "Room 0".
Understand "Area 31 in Room 0" as a31r0.
the parent of the a31r0 is r0.
the X of the a31r0 is "545153.5".
the Y of the a31r0 is "1352938.00390625".

[create a42r0]
a42r0 is a area. "An area (42) in r0".
the printed name of the a42r0 is "Room 0".
Understand "Area 42 in Room 0" as a42r0.
the parent of the a42r0 is r0.
the X of the a42r0 is "544948.0".
the Y of the a42r0 is "1353139.0".

[create a43r0]
a43r0 is a area. "An area (43) in r0".
the printed name of the a43r0 is "Room 0".
Understand "Area 43 in Room 0" as a43r0.
the parent of the a43r0 is r0.
the X of the a43r0 is "544946.6865942029".
the Y of the a43r0 is "1353089.6751110333".

[create a49r0]
a49r0 is a area. "An area (49) in r0".
the printed name of the a49r0 is "Room 0".
Understand "Area 49 in Room 0" as a49r0.
the parent of the a49r0 is r0.
the X of the a49r0 is "546412.8094615669".
the Y of the a49r0 is "1353444.8713538798".

[create a56r0]
a56r0 is a area. "An area (56) in r0".
the printed name of the a56r0 is "Room 0".
Understand "Area 56 in Room 0" as a56r0.
the parent of the a56r0 is r0.
the X of the a56r0 is "545141.5483636889".
the Y of the a56r0 is "1352966.9119265778".

[create a59r0]
a59r0 is a area. "An area (59) in r0".
the printed name of the a59r0 is "Room 0".
Understand "Area 59 in Room 0" as a59r0.
the parent of the a59r0 is r0.
the X of the a59r0 is "545107.9912057454".
the Y of the a59r0 is "1352964.2196130904".

[create a71r0]
a71r0 is a area. "An area (71) in r0".
the printed name of the a71r0 is "Room 0".
Understand "Area 71 in Room 0" as a71r0.
the parent of the a71r0 is r0.
the X of the a71r0 is "545545.7188426536".
the Y of the a71r0 is "1353465.5777999226".

[create a72r0]
a72r0 is a area. "An area (72) in r0".
the printed name of the a72r0 is "Room 0".
Understand "Area 72 in Room 0" as a72r0.
the parent of the a72r0 is r0.
the X of the a72r0 is "546035.5".
the Y of the a72r0 is "1353572.0990566039".

[create a0r1]
a0r1 is a area. "An area (0) in r1".
the printed name of the a0r1 is "Room 1".
Understand "Area 0 in Room 1" as a0r1.
the parent of the a0r1 is r1.
the X of the a0r1 is "547177.0".
the Y of the a0r1 is "1355222.0".

[create a1r1]
a1r1 is a area. "An area (1) in r1".
the printed name of the a1r1 is "Room 1".
Understand "Area 1 in Room 1" as a1r1.
the parent of the a1r1 is r1.
the X of the a1r1 is "547098.1021505377".
the Y of the a1r1 is "1355187.1666666665".

[create a2r1]
a2r1 is a area. "An area (2) in r1".
the printed name of the a2r1 is "Room 1".
Understand "Area 2 in Room 1" as a2r1.
the parent of the a2r1 is r1.
the X of the a2r1 is "545175.0".
the Y of the a2r1 is "1355424.0".

[create a3r1]
a3r1 is a area. "An area (3) in r1".
the printed name of the a3r1 is "Room 1".
Understand "Area 3 in Room 1" as a3r1.
the parent of the a3r1 is r1.
the X of the a3r1 is "545217.2698675498".
the Y of the a3r1 is "1355442.4980132452".

[create a6r1]
a6r1 is a area. "An area (6) in r1".
the printed name of the a6r1 is "Room 1".
Understand "Area 6 in Room 1" as a6r1.
the parent of the a6r1 is r1.
the X of the a6r1 is "545184.0".
the Y of the a6r1 is "1355514.0".

[create a7r1]
a7r1 is a area. "An area (7) in r1".
the printed name of the a7r1 is "Room 1".
Understand "Area 7 in Room 1" as a7r1.
the parent of the a7r1 is r1.
the X of the a7r1 is "545215.0876068377".
the Y of the a7r1 is "1355488.1662393163".

[create a8r1]
a8r1 is a area. "An area (8) in r1".
the printed name of the a8r1 is "Room 1".
Understand "Area 8 in Room 1" as a8r1.
the parent of the a8r1 is r1.
the X of the a8r1 is "545996.1111111111".
the Y of the a8r1 is "1355504.0".

[create a9r1]
a9r1 is a area. "An area (9) in r1".
the printed name of the a9r1 is "Room 1".
Understand "Area 9 in Room 1" as a9r1.
the parent of the a9r1 is r1.
the X of the a9r1 is "545964.6047004999".
the Y of the a9r1 is "1355505.5610161046".

[create a10r1]
a10r1 is a area. "An area (10) in r1".
the printed name of the a10r1 is "Room 1".
Understand "Area 10 in Room 1" as a10r1.
the parent of the a10r1 is r1.
the X of the a10r1 is "545626.0".
the Y of the a10r1 is "1355514.0".

[create a11r1]
a11r1 is a area. "An area (11) in r1".
the printed name of the a11r1 is "Room 1".
Understand "Area 11 in Room 1" as a11r1.
the parent of the a11r1 is r1.
the X of the a11r1 is "545656.4133125902".
the Y of the a11r1 is "1355505.9086687411".

[create a26r1]
a26r1 is a area. "An area (26) in r1".
the printed name of the a26r1 is "Room 1".
Understand "Area 26 in Room 1" as a26r1.
the parent of the a26r1 is r1.
the X of the a26r1 is "547177.0".
the Y of the a26r1 is "1354638.0".

[create a27r1]
a27r1 is a area. "An area (27) in r1".
the printed name of the a27r1 is "Room 1".
Understand "Area 27 in Room 1" as a27r1.
the parent of the a27r1 is r1.
the X of the a27r1 is "547115.2809236952".
the Y of the a27r1 is "1354675.5".

[create a53r1]
a53r1 is a area. "An area (53) in r1".
the printed name of the a53r1 is "Room 1".
Understand "Area 53 in Room 1" as a53r1.
the parent of the a53r1 is r1.
the X of the a53r1 is "547036.6850543967".
the Y of the a53r1 is "1354725.2214721933".

[create a54r1]
a54r1 is a area. "An area (54) in r1".
the printed name of the a54r1 is "Room 1".
Understand "Area 54 in Room 1" as a54r1.
the parent of the a54r1 is r1.
the X of the a54r1 is "547037.6040031037".
the Y of the a54r1 is "1355160.374629946".

[create a55r1]
a55r1 is a area. "An area (55) in r1".
the printed name of the a55r1 is "Room 1".
Understand "Area 55 in Room 1" as a55r1.
the parent of the a55r1 is r1.
the X of the a55r1 is "545232.1884979302".
the Y of the a55r1 is "1355459.7787601028".

[create a61r1]
a61r1 is a area. "An area (61) in r1".
the printed name of the a61r1 is "Room 1".
Understand "Area 61 in Room 1" as a61r1.
the parent of the a61r1 is r1.
the X of the a61r1 is "545953.6461810255".
the Y of the a61r1 is "1355404.85976856".

[create a62r1]
a62r1 is a area. "An area (62) in r1".
the printed name of the a62r1 is "Room 1".
Understand "Area 62 in Room 1" as a62r1.
the parent of the a62r1 is r1.
the X of the a62r1 is "545666.922122437".
the Y of the a62r1 is "1355409.5113160554".

[create a63r1]
a63r1 is a area. "An area (63) in r1".
the printed name of the a63r1 is "Room 1".
Understand "Area 63 in Room 1" as a63r1.
the parent of the a63r1 is r1.
the X of the a63r1 is "546032.3445626477".
the Y of the a63r1 is "1355334.3503250591".

[create a64r1]
a64r1 is a area. "An area (64) in r1".
the printed name of the a64r1 is "Room 1".
Understand "Area 64 in Room 1" as a64r1.
the parent of the a64r1 is r1.
the X of the a64r1 is "545585.5".
the Y of the a64r1 is "1355340.855168269".

[create a68r1]
a68r1 is a area. "An area (68) in r1".
the printed name of the a68r1 is "Room 1".
Understand "Area 68 in Room 1" as a68r1.
the parent of the a68r1 is r1.
the X of the a68r1 is "546037.2905415163".
the Y of the a68r1 is "1354443.0".

[create a69r1]
a69r1 is a area. "An area (69) in r1".
the printed name of the a69r1 is "Room 1".
Understand "Area 69 in Room 1" as a69r1.
the parent of the a69r1 is r1.
the X of the a69r1 is "546633.1538537387".
the Y of the a69r1 is "1354395.928650952".

[create a14r2]
a14r2 is a area. "An area (14) in r2".
the printed name of the a14r2 is "Room 2".
Understand "Area 14 in Room 2" as a14r2.
the parent of the a14r2 is r2.
the X of the a14r2 is "549001.0".
the Y of the a14r2 is "1353480.0".

[create a15r2]
a15r2 is a area. "An area (15) in r2".
the printed name of the a15r2 is "Room 2".
Understand "Area 15 in Room 2" as a15r2.
the parent of the a15r2 is r2.
the X of the a15r2 is "548961.3329344185".
the Y of the a15r2 is "1353461.533748205".

[create a16r2]
a16r2 is a area. "An area (16) in r2".
the printed name of the a16r2 is "Room 2".
Understand "Area 16 in Room 2" as a16r2.
the parent of the a16r2 is r2.
the X of the a16r2 is "547646.0".
the Y of the a16r2 is "1353480.0".

[create a17r2]
a17r2 is a area. "An area (17) in r2".
the printed name of the a17r2 is "Room 2".
Understand "Area 17 in Room 2" as a17r2.
the parent of the a17r2 is r2.
the X of the a17r2 is "547605.7054722286".
the Y of the a17r2 is "1353461.5971994374".

[create a18r2]
a18r2 is a area. "An area (18) in r2".
the printed name of the a18r2 is "Room 2".
Understand "Area 18 in Room 2" as a18r2.
the parent of the a18r2 is r2.
the X of the a18r2 is "548141.0".
the Y of the a18r2 is "1353480.0".

[create a19r2]
a19r2 is a area. "An area (19) in r2".
the printed name of the a19r2 is "Room 2".
Understand "Area 19 in Room 2" as a19r2.
the parent of the a19r2 is r2.
the X of the a19r2 is "548182.3556858186".
the Y of the a19r2 is "1353461.7045075546".

[create a20r2]
a20r2 is a area. "An area (20) in r2".
the printed name of the a20r2 is "Room 2".
Understand "Area 20 in Room 2" as a20r2.
the parent of the a20r2 is r2.
the X of the a20r2 is "547549.0".
the Y of the a20r2 is "1353391.0".

[create a21r2]
a21r2 is a area. "An area (21) in r2".
the printed name of the a21r2 is "Room 2".
Understand "Area 21 in Room 2" as a21r2.
the parent of the a21r2 is r2.
the X of the a21r2 is "547511.6655360125".
the Y of the a21r2 is "1353396.827877191".

[create a24r2]
a24r2 is a area. "An area (24) in r2".
the printed name of the a24r2 is "Room 2".
Understand "Area 24 in Room 2" as a24r2.
the parent of the a24r2 is r2.
the X of the a24r2 is "547030.0".
the Y of the a24r2 is "1353372.0".

[create a25r2]
a25r2 is a area. "An area (25) in r2".
the printed name of the a25r2 is "Room 2".
Understand "Area 25 in Room 2" as a25r2.
the parent of the a25r2 is r2.
the X of the a25r2 is "547065.8212004292".
the Y of the a25r2 is "1353407.5081115812".

[create a32r2]
a32r2 is a area. "An area (32) in r2".
the printed name of the a32r2 is "Room 2".
Understand "Area 32 in Room 2" as a32r2.
the parent of the a32r2 is r2.
the X of the a32r2 is "546878.0".
the Y of the a32r2 is "1353391.0".

[create a33r2]
a33r2 is a area. "An area (33) in r2".
the printed name of the a33r2 is "Room 2".
Understand "Area 33 in Room 2" as a33r2.
the parent of the a33r2 is r2.
the X of the a33r2 is "546911.4752246249".
the Y of the a33r2 is "1353416.8626631643".

[create a34r2]
a34r2 is a area. "An area (34) in r2".
the printed name of the a34r2 is "Room 2".
Understand "Area 34 in Room 2" as a34r2.
the parent of the a34r2 is r2.
the X of the a34r2 is "546869.0".
the Y of the a34r2 is "1353480.0".

[create a35r2]
a35r2 is a area. "An area (35) in r2".
the printed name of the a35r2 is "Room 2".
Understand "Area 35 in Room 2" as a35r2.
the parent of the a35r2 is r2.
the X of the a35r2 is "546897.1477777245".
the Y of the a35r2 is "1353460.368876399".

[create a36r2]
a36r2 is a area. "An area (36) in r2".
the printed name of the a36r2 is "Room 2".
Understand "Area 36 in Room 2" as a36r2.
the parent of the a36r2 is r2.
the X of the a36r2 is "548992.0".
the Y of the a36r2 is "1353391.0".

[create a37r2]
a37r2 is a area. "An area (37) in r2".
the printed name of the a37r2 is "Room 2".
Understand "Area 37 in Room 2" as a37r2.
the parent of the a37r2 is r2.
the X of the a37r2 is "548960.7862838916".
the Y of the a37r2 is "1353416.633971292".

[create a38r2]
a38r2 is a area. "An area (38) in r2".
the printed name of the a38r2 is "Room 2".
Understand "Area 38 in Room 2" as a38r2.
the parent of the a38r2 is r2.
the X of the a38r2 is "547637.0".
the Y of the a38r2 is "1353391.0".

[create a39r2]
a39r2 is a area. "An area (39) in r2".
the printed name of the a39r2 is "Room 2".
Understand "Area 39 in Room 2" as a39r2.
the parent of the a39r2 is r2.
the X of the a39r2 is "547616.1959693881".
the Y of the a39r2 is "1353415.5813064666".

[create a40r2]
a40r2 is a area. "An area (40) in r2".
the printed name of the a40r2 is "Room 2".
Understand "Area 40 in Room 2" as a40r2.
the parent of the a40r2 is r2.
the X of the a40r2 is "548150.0".
the Y of the a40r2 is "1353391.0".

[create a41r2]
a41r2 is a area. "An area (41) in r2".
the printed name of the a41r2 is "Room 2".
Understand "Area 41 in Room 2" as a41r2.
the parent of the a41r2 is r2.
the X of the a41r2 is "548181.2137161086".
the Y of the a41r2 is "1353416.633971292".

[create a46r2]
a46r2 is a area. "An area (46) in r2".
the printed name of the a46r2 is "Room 2".
Understand "Area 46 in Room 2" as a46r2.
the parent of the a46r2 is r2.
the X of the a46r2 is "547590.6590073579".
the Y of the a46r2 is "1353444.6217006089".

[create a47r2]
a47r2 is a area. "An area (47) in r2".
the printed name of the a47r2 is "Room 2".
Understand "Area 47 in Room 2" as a47r2.
the parent of the a47r2 is r2.
the X of the a47r2 is "548197.6344696968".
the Y of the a47r2 is "1353444.0018939392".

[create a48r2]
a48r2 is a area. "An area (48) in r2".
the printed name of the a48r2 is "Room 2".
Understand "Area 48 in Room 2" as a48r2.
the parent of the a48r2 is r2.
the X of the a48r2 is "547507.4004165076".
the Y of the a48r2 is "1353443.7436216874".

[create a50r2]
a50r2 is a area. "An area (50) in r2".
the printed name of the a50r2 is "Room 2".
Understand "Area 50 in Room 2" as a50r2.
the parent of the a50r2 is r2.
the X of the a50r2 is "547090.2615898387".
the Y of the a50r2 is "1353501.9488871447".

[create a51r2]
a51r2 is a area. "An area (51) in r2".
the printed name of the a51r2 is "Room 2".
Understand "Area 51 in Room 2" as a51r2.
the parent of the a51r2 is r2.
the X of the a51r2 is "547137.1997386224".
the Y of the a51r2 is "1353557.282386464".

[create a52r2]
a52r2 is a area. "An area (52) in r2".
the printed name of the a52r2 is "Room 2".
Understand "Area 52 in Room 2" as a52r2.
the parent of the a52r2 is r2.
the X of the a52r2 is "547201.0047352596".
the Y of the a52r2 is "1353556.9922982883".

[create a57r2]
a57r2 is a area. "An area (57) in r2".
the printed name of the a57r2 is "Room 2".
Understand "Area 57 in Room 2" as a57r2.
the parent of the a57r2 is r2.
the X of the a57r2 is "546924.0367085102".
the Y of the a57r2 is "1353439.284650326".

[create a58r2]
a58r2 is a area. "An area (58) in r2".
the printed name of the a58r2 is "Room 2".
Understand "Area 58 in Room 2" as a58r2.
the parent of the a58r2 is r2.
the X of the a58r2 is "548944.7901234571".
the Y of the a58r2 is "1353443.2942386835".

[create a70r2]
a70r2 is a area. "An area (70) in r2".
the printed name of the a70r2 is "Room 2".
Understand "Area 70 in Room 2" as a70r2.
the parent of the a70r2 is r2.
the X of the a70r2 is "548578.4310969368".
the Y of the a70r2 is "1353567.4370029136".

[create a4r3]
a4r3 is a area. "An area (4) in r3".
the printed name of the a4r3 is "Room 3".
Understand "Area 4 in Room 3" as a4r3.
the parent of the a4r3 is r3.
the X of the a4r3 is "547757.0".
the Y of the a4r3 is "1355501.0".

[create a5r3]
a5r3 is a area. "An area (5) in r3".
the printed name of the a5r3 is "Room 3".
Understand "Area 5 in Room 3" as a5r3.
the parent of the a5r3 is r3.
the X of the a5r3 is "547797.0707313817".
the Y of the a5r3 is "1355447.2053135843".

[create a44r3]
a44r3 is a area. "An area (44) in r3".
the printed name of the a44r3 is "Room 3".
Understand "Area 44 in Room 3" as a44r3.
the parent of the a44r3 is r3.
the X of the a44r3 is "548024.0".
the Y of the a44r3 is "1355532.0".

[create a45r3]
a45r3 is a area. "An area (45) in r3".
the printed name of the a45r3 is "Room 3".
Understand "Area 45 in Room 3" as a45r3.
the parent of the a45r3 is r3.
the X of the a45r3 is "547996.2593157241".
the Y of the a45r3 is "1355479.5971839246".

[create a60r3]
a60r3 is a area. "An area (60) in r3".
the printed name of the a60r3 is "Room 3".
Understand "Area 60 in Room 3" as a60r3.
the parent of the a60r3 is r3.
the X of the a60r3 is "547871.3663274824".
the Y of the a60r3 is "1355364.1920982176".

[create a65r3]
a65r3 is a area. "An area (65) in r3".
the printed name of the a65r3 is "Room 3".
Understand "Area 65 in Room 3" as a65r3.
the parent of the a65r3 is r3.
the X of the a65r3 is "547923.7557274037".
the Y of the a65r3 is "1355363.3385288815".

[create a66r3]
a66r3 is a area. "An area (66) in r3".
the printed name of the a66r3 is "Room 3".
Understand "Area 66 in Room 3" as a66r3.
the parent of the a66r3 is r3.
the X of the a66r3 is "547473.0802226814".
the Y of the a66r3 is "1354243.1975382022".

[create a67r3]
a67r3 is a area. "An area (67) in r3".
the printed name of the a67r3 is "Room 3".
Understand "Area 67 in Room 3" as a67r3.
the parent of the a67r3 is r3.
the X of the a67r3 is "548341.9076679607".
the Y of the a67r3 is "1354256.9294065498".

[create d0]
d0 is a door. "A door between a71r0 and a64r1".
d0 is north of a71r0 and south of a64r1.

[create d1]
d1 is a door. "A door between a72r0 and a68r1".
d1 is north of a72r0 and south of a68r1.

[create d2]
d2 is a door. "A door between a52r2 and a66r3".
d2 is north of a52r2 and south of a66r3.

[create d3]
d3 is a door. "A door between a70r2 and a67r3".
d3 is north of a70r2 and south of a67r3.

[create d6]
d6 is a door. "A door between a69r1 and a51r2".
d6 is southeast of a69r1 and northwest of a51r2.

southwest of a0r1 is northeast of a1r1.

northeast of a2r1 is southwest of a3r1.

southeast of a4r3 is northwest of a5r3.

southeast of a6r1 is northwest of a7r1.

west of a8r1 is east of a9r1.

east of a10r1 is west of a11r1.

northwest of a12r0 is southeast of a13r0.

southwest of a14r2 is northeast of a15r2.

southwest of a16r2 is northeast of a17r2.

southeast of a18r2 is northwest of a19r2.

west of a20r2 is east of a21r2.

west of a22r0 is east of a23r0.

northeast of a24r2 is southwest of a25r2.

northwest of a26r1 is southeast of a27r1.

north of a28r0 is south of a29r0.

north of a30r0 is south of a31r0.

northeast of a32r2 is southwest of a33r2.

southeast of a34r2 is northwest of a35r2.

northwest of a36r2 is southeast of a37r2.

northwest of a38r2 is southeast of a39r2.

northeast of a40r2 is southwest of a41r2.

south of a42r0 is north of a43r0.

southwest of a44r3 is northeast of a45r3.

southwest of a17r2 is northeast of a46r2.

southeast of a19r2 is northwest of a47r2.

north of a21r2 is south of a48r2.

southwest of a23r0 is northeast of a49r0.

north of a25r2 is south of a50r2.

east of a51r2 is west of a52r2.

northwest of a27r1 is southeast of a53r1.

southwest of a1r1 is northeast of a54r1.

northeast of a3r1 is southwest of a55r1.

north of a31r0 is south of a56r0.

northeast of a33r2 is southwest of a57r2.

southeast of a35r2 is northwest of a57r2.

northwest of a37r2 is southeast of a58r2.

northwest of a39r2 is southeast of a46r2.

northeast of a41r2 is southwest of a47r2.

northwest of a13r0 is southeast of a49r0.

southwest of a15r2 is northeast of a58r2.

northeast of a29r0 is southwest of a59r0.

southeast of a5r3 is northwest of a60r3.

southeast of a7r1 is northwest of a55r1.

south of a9r1 is north of a61r1.

south of a11r1 is north of a62r1.

southeast of a61r1 is northwest of a63r1.

southwest of a62r1 is northeast of a64r1.

east of a55r1 is west of a64r1.

west of a61r1 is east of a62r1.

east of a60r3 is west of a65r3.

southwest of a45r3 is northeast of a65r3.

east of a66r3 is west of a67r3.

south of a60r3 is north of a66r3.

south of a65r3 is north of a67r3.

east of a68r1 is west of a69r1.

southwest of a53r1 is northeast of a69r1.

south of a63r1 is north of a68r1.

north of a53r1 is south of a54r1.

west of a54r1 is east of a63r1.

west of a58r2 is east of a70r2.

west of a48r2 is east of a52r2.

northeast of a50r2 is southwest of a51r2.

west of a56r0 is east of a59r0.

northeast of a56r0 is southwest of a71r0.

southeast of a43r0 is northwest of a59r0.

west of a46r2 is east of a48r2.

west of a50r2 is east of a57r2.

east of a47r2 is west of a70r2.

east of a71r0 is west of a72r0.

west of a49r0 is east of a72r0.

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
the player is in a53r1.

the orientation of the player is 6.
