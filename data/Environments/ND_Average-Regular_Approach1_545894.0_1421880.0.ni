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

[create a0r0]
a0r0 is a area. "An area (0) in r0".
the printed name of the a0r0 is "Room 0".
Understand "Area 0 in Room 0" as a0r0.
the parent of the a0r0 is r0.
the X of the a0r0 is "547200.9215804252".
the Y of the a0r0 is "1422558.9999999998".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "547200.9993983153".
the Y of the a1r0 is "1422632.3333333333".

[create a2r0]
a2r0 is a area. "An area (2) in r0".
the printed name of the a2r0 is "Room 0".
Understand "Area 2 in Room 0" as a2r0.
the parent of the a2r0 is r0.
the X of the a2r0 is "547200.6028880867".
the Y of the a2r0 is "1422707.55".

[create a4r0]
a4r0 is a area. "An area (4) in r0".
the printed name of the a4r0 is "Room 0".
Understand "Area 4 in Room 0" as a4r0.
the parent of the a4r0 is r0.
the X of the a4r0 is "546968.700326087".
the Y of the a4r0 is "1422780.2".

[create a5r0]
a5r0 is a area. "An area (5) in r0".
the printed name of the a5r0 is "Room 0".
Understand "Area 5 in Room 0" as a5r0.
the parent of the a5r0 is r0.
the X of the a5r0 is "547200.268267148".
the Y of the a5r0 is "1422784.65".

[create a10r0]
a10r0 is a area. "An area (10) in r0".
the printed name of the a10r0 is "Room 0".
Understand "Area 10 in Room 0" as a10r0.
the parent of the a10r0 is r0.
the X of the a10r0 is "546966.4220652173".
the Y of the a10r0 is "1422489.0".

[create a11r0]
a11r0 is a area. "An area (11) in r0".
the printed name of the a11r0 is "Room 0".
Understand "Area 11 in Room 0" as a11r0.
the parent of the a11r0 is r0.
the X of the a11r0 is "546965.6351086957".
the Y of the a11r0 is "1422416.2".

[create a12r0]
a12r0 is a area. "An area (12) in r0".
the printed name of the a12r0 is "Room 0".
Understand "Area 12 in Room 0" as a12r0.
the parent of the a12r0 is r0.
the X of the a12r0 is "547200.6977537104".
the Y of the a12r0 is "1422339.0".

[create a13r0]
a13r0 is a area. "An area (13) in r0".
the printed name of the a13r0 is "Room 0".
Understand "Area 13 in Room 0" as a13r0.
the parent of the a13r0 is r0.
the X of the a13r0 is "547200.7707581227".
the Y of the a13r0 is "1422412.3333333335".

[create a14r0]
a14r0 is a area. "An area (14) in r0".
the printed name of the a14r0 is "Room 0".
Understand "Area 14 in Room 0" as a14r0.
the parent of the a14r0 is r0.
the X of the a14r0 is "546967.1220652175".
the Y of the a14r0 is "1422561.8".

[create a15r0]
a15r0 is a area. "An area (15) in r0".
the printed name of the a15r0 is "Room 0".
Understand "Area 15 in Room 0" as a15r0.
the parent of the a15r0 is r0.
the X of the a15r0 is "547200.8453670278".
the Y of the a15r0 is "1422485.6666666665".

[create a16r0]
a16r0 is a area. "An area (16) in r0".
the printed name of the a16r0 is "Room 0".
Understand "Area 16 in Room 0" as a16r0.
the parent of the a16r0 is r0.
the X of the a16r0 is "546967.7351086956".
the Y of the a16r0 is "1422634.6".

[create a17r0]
a17r0 is a area. "An area (17) in r0".
the printed name of the a17r0 is "Room 0".
Understand "Area 17 in Room 0" as a17r0.
the parent of the a17r0 is r0.
the X of the a17r0 is "548703.0324175663".
the Y of the a17r0 is "1422502.0135168608".

[create a19r0]
a19r0 is a area. "An area (19) in r0".
the printed name of the a19r0 is "Room 0".
Understand "Area 19 in Room 0" as a19r0.
the parent of the a19r0 is r0.
the X of the a19r0 is "546891.3237452032".
the Y of the a19r0 is "1421906.3646405656".

[create a20r0]
a20r0 is a area. "An area (20) in r0".
the printed name of the a20r0 is "Room 0".
Understand "Area 20 in Room 0" as a20r0.
the parent of the a20r0 is r0.
the X of the a20r0 is "546945.7779347856".
the Y of the a20r0 is "1422002.5596938413".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "546888.8657919269".
the Y of the a22r0 is "1422020.4354443646".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "546989.3517621404".
the Y of the a23r0 is "1422015.2785383726".

[create a24r0]
a24r0 is a area. "An area (24) in r0".
the printed name of the a24r0 is "Room 0".
Understand "Area 24 in Room 0" as a24r0.
the parent of the a24r0 is r0.
the X of the a24r0 is "547144.2215655078".
the Y of the a24r0 is "1422022.1562713669".

[create a25r0]
a25r0 is a area. "An area (25) in r0".
the printed name of the a25r0 is "Room 0".
Understand "Area 25 in Room 0" as a25r0.
the parent of the a25r0 is r0.
the X of the a25r0 is "546969.3445136291".
the Y of the a25r0 is "1422262.1566995915".

[create a26r0]
a26r0 is a area. "An area (26) in r0".
the printed name of the a26r0 is "Room 0".
Understand "Area 26 in Room 0" as a26r0.
the parent of the a26r0 is r0.
the X of the a26r0 is "546967.1724611709".
the Y of the a26r0 is "1422343.4".

[create a30r0]
a30r0 is a area. "An area (30) in r0".
the printed name of the a30r0 is "Room 0".
Understand "Area 30 in Room 0" as a30r0.
the parent of the a30r0 is r0.
the X of the a30r0 is "548222.4184328839".
the Y of the a30r0 is "1421936.1969348688".

[create a48r0]
a48r0 is a area. "An area (48) in r0".
the printed name of the a48r0 is "Room 0".
Understand "Area 48 in Room 0" as a48r0.
the parent of the a48r0 is r0.
the X of the a48r0 is "546846.0".
the Y of the a48r0 is "1421772.0".

[create a49r0]
a49r0 is a area. "An area (49) in r0".
the printed name of the a49r0 is "Room 0".
Understand "Area 49 in Room 0" as a49r0.
the parent of the a49r0 is r0.
the X of the a49r0 is "546872.9601572268".
the Y of the a49r0 is "1421808.7964757935".

[create a50r0]
a50r0 is a area. "An area (50) in r0".
the printed name of the a50r0 is "Room 0".
Understand "Area 50 in Room 0" as a50r0.
the parent of the a50r0 is r0.
the X of the a50r0 is "548280.0".
the Y of the a50r0 is "1421800.0".

[create a51r0]
a51r0 is a area. "An area (51) in r0".
the printed name of the a51r0 is "Room 0".
Understand "Area 51 in Room 0" as a51r0.
the parent of the a51r0 is r0.
the X of the a51r0 is "548243.875".
the Y of the a51r0 is "1421885.7209595959".

[create a52r0]
a52r0 is a area. "An area (52) in r0".
the printed name of the a52r0 is "Room 0".
Understand "Area 52 in Room 0" as a52r0.
the parent of the a52r0 is r0.
the X of the a52r0 is "546765.645455358".
the Y of the a52r0 is "1421880.1274465993".

[create a53r0]
a53r0 is a area. "An area (53) in r0".
the printed name of the a53r0 is "Room 0".
Understand "Area 53 in Room 0" as a53r0.
the parent of the a53r0 is r0.
the X of the a53r0 is "546756.9611690349".
the Y of the a53r0 is "1421922.4109719293".

[create a54r0]
a54r0 is a area. "An area (54) in r0".
the printed name of the a54r0 is "Room 0".
Understand "Area 54 in Room 0" as a54r0.
the parent of the a54r0 is r0.
the X of the a54r0 is "548888.0".
the Y of the a54r0 is "1422427.0".

[create a55r0]
a55r0 is a area. "An area (55) in r0".
the printed name of the a55r0 is "Room 0".
Understand "Area 55 in Room 0" as a55r0.
the parent of the a55r0 is r0.
the X of the a55r0 is "548799.3368675596".
the Y of the a55r0 is "1422462.2857142857".

[create a74r0]
a74r0 is a area. "An area (74) in r0".
the printed name of the a74r0 is "Room 0".
Understand "Area 74 in Room 0" as a74r0.
the parent of the a74r0 is r0.
the X of the a74r0 is "546765.0".
the Y of the a74r0 is "1421880.0".

[create a75r0]
a75r0 is a area. "An area (75) in r0".
the printed name of the a75r0 is "Room 0".
Understand "Area 75 in Room 0" as a75r0.
the parent of the a75r0 is r0.
the X of the a75r0 is "546769.6801000858".
the Y of the a75r0 is "1421857.995740458".

[create a76r0]
a76r0 is a area. "An area (76) in r0".
the printed name of the a76r0 is "Room 0".
Understand "Area 76 in Room 0" as a76r0.
the parent of the a76r0 is r0.
the X of the a76r0 is "546829.9512464551".
the Y of the a76r0 is "1421793.5956763818".

[create a77r0]
a77r0 is a area. "An area (77) in r0".
the printed name of the a77r0 is "Room 0".
Understand "Area 77 in Room 0" as a77r0.
the parent of the a77r0 is r0.
the X of the a77r0 is "546800.8670190257".
the Y of the a77r0 is "1421816.1944176543".

[create a88r0]
a88r0 is a area. "An area (88) in r0".
the printed name of the a88r0 is "Room 0".
Understand "Area 88 in Room 0" as a88r0.
the parent of the a88r0 is r0.
the X of the a88r0 is "546833.674865904".
the Y of the a88r0 is "1421863.5230717422".

[create a3r1]
a3r1 is a area. "An area (3) in r1".
the printed name of the a3r1 is "Room 1".
Understand "Area 3 in Room 1" as a3r1.
the parent of the a3r1 is r1.
the X of the a3r1 is "546969.1490123457".
the Y of the a3r1 is "1422853.0".

[create a6r1]
a6r1 is a area. "An area (6) in r1".
the printed name of the a6r1 is "Room 1".
Understand "Area 6 in Room 1" as a6r1.
the parent of the a6r1 is r1.
the X of the a6r1 is "546969.4443947934".
the Y of the a6r1 is "1422925.8".

[create a7r1]
a7r1 is a area. "An area (7) in r1".
the printed name of the a7r1 is "Room 1".
Understand "Area 7 in Room 1" as a7r1.
the parent of the a7r1 is r1.
the X of the a7r1 is "547199.9683393502".
the Y of the a7r1 is "1422861.75".

[create a8r1]
a8r1 is a area. "An area (8) in r1".
the printed name of the a8r1 is "Room 1".
Understand "Area 8 in Room 1" as a8r1.
the parent of the a8r1 is r1.
the X of the a8r1 is "546969.58647343".
the Y of the a8r1 is "1422998.6".

[create a9r1]
a9r1 is a area. "An area (9) in r1".
the printed name of the a9r1 is "Room 1".
Understand "Area 9 in Room 1" as a9r1.
the parent of the a9r1 is r1.
the X of the a9r1 is "547199.7031046932".
the Y of the a9r1 is "1422938.8499999999".

[create a36r1]
a36r1 is a area. "An area (36) in r1".
the printed name of the a36r1 is "Room 1".
Understand "Area 36 in Room 1" as a36r1.
the parent of the a36r1 is r1.
the X of the a36r1 is "547289.7990288597".
the Y of the a36r1 is "1423805.2121009345".

[create a37r1]
a37r1 is a area. "An area (37) in r1".
the printed name of the a37r1 is "Room 1".
Understand "Area 37 in Room 1" as a37r1.
the parent of the a37r1 is r1.
the X of the a37r1 is "548335.0944238".
the Y of the a37r1 is "1423783.5825551737".

[create a38r1]
a38r1 is a area. "An area (38) in r1".
the printed name of the a38r1 is "Room 1".
Understand "Area 38 in Room 1" as a38r1.
the parent of the a38r1 is r1.
the X of the a38r1 is "546612.4446009535".
the Y of the a38r1 is "1423864.1811858914".

[create a39r1]
a39r1 is a area. "An area (39) in r1".
the printed name of the a39r1 is "Room 1".
Understand "Area 39 in Room 1" as a39r1.
the parent of the a39r1 is r1.
the X of the a39r1 is "546834.4738955817".
the Y of the a39r1 is "1423840.6224899597".

[create a40r1]
a40r1 is a area. "An area (40) in r1".
the printed name of the a40r1 is "Room 1".
Understand "Area 40 in Room 1" as a40r1.
the parent of the a40r1 is r1.
the X of the a40r1 is "547051.5873019709".
the Y of the a40r1 is "1423698.6504095348".

[create a41r1]
a41r1 is a area. "An area (41) in r1".
the printed name of the a41r1 is "Room 1".
Understand "Area 41 in Room 1" as a41r1.
the parent of the a41r1 is r1.
the X of the a41r1 is "547150.475287367".
the Y of the a41r1 is "1423688.3122290676".

[create a42r1]
a42r1 is a area. "An area (42) in r1".
the printed name of the a42r1 is "Room 1".
Understand "Area 42 in Room 1" as a42r1.
the parent of the a42r1 is r1.
the X of the a42r1 is "547199.4725631766".
the Y of the a42r1 is "1423015.9500000002".

[create a43r1]
a43r1 is a area. "An area (43) in r1".
the printed name of the a43r1 is "Room 1".
Understand "Area 43 in Room 1" as a43r1.
the parent of the a43r1 is r1.
the X of the a43r1 is "547198.9962093864".
the Y of the a43r1 is "1423093.05".

[create a44r1]
a44r1 is a area. "An area (44) in r1".
the printed name of the a44r1 is "Room 1".
Understand "Area 44 in Room 1" as a44r1.
the parent of the a44r1 is r1.
the X of the a44r1 is "547198.8188086643".
the Y of the a44r1 is "1423170.15".

[create a45r1]
a45r1 is a area. "An area (45) in r1".
the printed name of the a45r1 is "Room 1".
Understand "Area 45 in Room 1" as a45r1.
the parent of the a45r1 is r1.
the X of the a45r1 is "547198.9403610106".
the Y of the a45r1 is "1423247.25".

[create a46r1]
a46r1 is a area. "An area (46) in r1".
the printed name of the a46r1 is "Room 1".
Understand "Area 46 in Room 1" as a46r1.
the parent of the a46r1 is r1.
the X of the a46r1 is "547199.360866426".
the Y of the a46r1 is "1423324.3499999999".

[create a47r1]
a47r1 is a area. "An area (47) in r1".
the printed name of the a47r1 is "Room 1".
Understand "Area 47 in Room 1" as a47r1.
the parent of the a47r1 is r1.
the X of the a47r1 is "547200.0803249099".
the Y of the a47r1 is "1423401.45".

[create a62r1]
a62r1 is a area. "An area (62) in r1".
the printed name of the a62r1 is "Room 1".
Understand "Area 62 in Room 1" as a62r1.
the parent of the a62r1 is r1.
the X of the a62r1 is "546558.0".
the Y of the a62r1 is "1423824.0".

[create a63r1]
a63r1 is a area. "An area (63) in r1".
the printed name of the a63r1 is "Room 1".
Understand "Area 63 in Room 1" as a63r1.
the parent of the a63r1 is r1.
the X of the a63r1 is "546585.2059471365".
the Y of the a63r1 is "1423844.0044052862".

[create a78r1]
a78r1 is a area. "An area (78) in r1".
the printed name of the a78r1 is "Room 1".
Understand "Area 78 in Room 1" as a78r1.
the parent of the a78r1 is r1.
the X of the a78r1 is "546567.0".
the Y of the a78r1 is "1423914.0".

[create a79r1]
a79r1 is a area. "An area (79) in r1".
the printed name of the a79r1 is "Room 1".
Understand "Area 79 in Room 1" as a79r1.
the parent of the a79r1 is r1.
the X of the a79r1 is "546597.9153780069".
the Y of the a79r1 is "1423888.1834621995".

[create a80r1]
a80r1 is a area. "An area (80) in r1".
the printed name of the a80r1 is "Room 1".
Understand "Area 80 in Room 1" as a80r1.
the parent of the a80r1 is r1.
the X of the a80r1 is "546796.0".
the Y of the a80r1 is "1423914.0".

[create a81r1]
a81r1 is a area. "An area (81) in r1".
the printed name of the a81r1 is "Room 1".
Understand "Area 81 in Room 1" as a81r1.
the parent of the a81r1 is r1.
the X of the a81r1 is "546827.8264604809".
the Y of the a81r1 is "1423905.767353952".

[create a82r1]
a82r1 is a area. "An area (82) in r1".
the printed name of the a82r1 is "Room 1".
Understand "Area 82 in Room 1" as a82r1.
the parent of the a82r1 is r1.
the X of the a82r1 is "547332.0".
the Y of the a82r1 is "1423914.0".

[create a83r1]
a83r1 is a area. "An area (83) in r1".
the printed name of the a83r1 is "Room 1".
Understand "Area 83 in Room 1" as a83r1.
the parent of the a83r1 is r1.
the X of the a83r1 is "547300.0059829061".
the Y of the a83r1 is "1423905.7505982907".

[create a86r1]
a86r1 is a area. "An area (86) in r1".
the printed name of the a86r1 is "Room 1".
Understand "Area 86 in Room 1" as a86r1.
the parent of the a86r1 is r1.
the X of the a86r1 is "548371.8043478262".
the Y of the a86r1 is "1423775.0".

[create a87r1]
a87r1 is a area. "An area (87) in r1".
the printed name of the a87r1 is "Room 1".
Understand "Area 87 in Room 1" as a87r1.
the parent of the a87r1 is r1.
the X of the a87r1 is "547185.814446161".
the Y of the a87r1 is "1423401.4499999997".

[create a18r2]
a18r2 is a area. "An area (18) in r2".
the printed name of the a18r2 is "Room 2".
Understand "Area 18 in Room 2" as a18r2.
the parent of the a18r2 is r2.
the X of the a18r2 is "548587.3786544112".
the Y of the a18r2 is "1423638.9435604704".

[create a84r2]
a84r2 is a area. "An area (84) in r2".
the printed name of the a84r2 is "Room 2".
Understand "Area 84 in Room 2" as a84r2.
the parent of the a84r2 is r2.
the X of the a84r2 is "548565.0".
the Y of the a84r2 is "1423675.677419355".

[create a21r3]
a21r3 is a area. "An area (21) in r3".
the printed name of the a21r3 is "Room 3".
Understand "Area 21 in Room 3" as a21r3.
the parent of the a21r3 is r3.
the X of the a21r3 is "545554.0365723589".
the Y of the a21r3 is "1423632.6014533546".

[create a31r3]
a31r3 is a area. "An area (31) in r3".
the printed name of the a31r3 is "Room 3".
Understand "Area 31 in Room 3" as a31r3.
the parent of the a31r3 is r3.
the X of the a31r3 is "545168.0926719012".
the Y of the a31r3 is "1423859.8065726864".

[create a32r3]
a32r3 is a area. "An area (32) in r3".
the printed name of the a32r3 is "Room 3".
Understand "Area 32 in Room 3" as a32r3.
the parent of the a32r3 is r3.
the X of the a32r3 is "545503.89814202".
the Y of the a32r3 is "1423756.089074554".

[create a58r3]
a58r3 is a area. "An area (58) in r3".
the printed name of the a58r3 is "Room 3".
Understand "Area 58 in Room 3" as a58r3.
the parent of the a58r3 is r3.
the X of the a58r3 is "545578.0".
the Y of the a58r3 is "1423904.0".

[create a59r3]
a59r3 is a area. "An area (59) in r3".
the printed name of the a59r3 is "Room 3".
Understand "Area 59 in Room 3" as a59r3.
the parent of the a59r3 is r3.
the X of the a59r3 is "545544.875".
the Y of the a59r3 is "1423830.0006598178".

[create a60r3]
a60r3 is a area. "An area (60) in r3".
the printed name of the a60r3 is "Room 3".
Understand "Area 60 in Room 3" as a60r3.
the parent of the a60r3 is r3.
the X of the a60r3 is "545111.0".
the Y of the a60r3 is "1423824.0".

[create a61r3]
a61r3 is a area. "An area (61) in r3".
the printed name of the a61r3 is "Room 3".
Understand "Area 61 in Room 3" as a61r3.
the parent of the a61r3 is r3.
the X of the a61r3 is "545153.06".
the Y of the a61r3 is "1423842.5189999999".

[create a64r3]
a64r3 is a area. "An area (64) in r3".
the printed name of the a64r3 is "Room 3".
Understand "Area 64 in Room 3" as a64r3.
the parent of the a64r3 is r3.
the X of the a64r3 is "545120.0".
the Y of the a64r3 is "1423914.0".

[create a65r3]
a65r3 is a area. "An area (65) in r3".
the printed name of the a65r3 is "Room 3".
Understand "Area 65 in Room 3" as a65r3.
the parent of the a65r3 is r3.
the X of the a65r3 is "545150.9153780071".
the Y of the a65r3 is "1423888.1834621993".

[create a27r4]
a27r4 is a area. "An area (27) in r4".
the printed name of the a27r4 is "Room 4".
Understand "Area 27 in Room 4" as a27r4.
the parent of the a27r4 is r4.
the X of the a27r4 is "545483.4387151967".
the Y of the a27r4 is "1421930.6903873074".

[create a28r4]
a28r4 is a area. "An area (28) in r4".
the printed name of the a28r4 is "Room 4".
Understand "Area 28 in Room 4" as a28r4.
the parent of the a28r4 is r4.
the X of the a28r4 is "545839.7586021506".
the Y of the a28r4 is "1421840.0134408604".

[create a29r4]
a29r4 is a area. "An area (29) in r4".
the printed name of the a29r4 is "Room 4".
Understand "Area 29 in Room 4" as a29r4.
the parent of the a29r4 is r4.
the X of the a29r4 is "545119.3147627445".
the Y of the a29r4 is "1421840.2898139763".

[create a66r4]
a66r4 is a area. "An area (66) in r4".
the printed name of the a66r4 is "Room 4".
Understand "Area 66 in Room 4" as a66r4.
the parent of the a66r4 is r4.
the X of the a66r4 is "545885.0".
the Y of the a66r4 is "1421791.0".

[create a67r4]
a67r4 is a area. "An area (67) in r4".
the printed name of the a67r4 is "Room 4".
Understand "Area 67 in Room 4" as a67r4.
the parent of the a67r4 is r4.
the X of the a67r4 is "545853.7862838916".
the Y of the a67r4 is "1421816.6339712918".

[create a68r4]
a68r4 is a area. "An area (68) in r4".
the printed name of the a68r4 is "Room 4".
Understand "Area 68 in Room 4" as a68r4.
the parent of the a68r4 is r4.
the X of the a68r4 is "545074.0".
the Y of the a68r4 is "1421791.0".

[create a69r4]
a69r4 is a area. "An area (69) in r4".
the printed name of the a69r4 is "Room 4".
Understand "Area 69 in Room 4" as a69r4.
the parent of the a69r4 is r4.
the X of the a69r4 is "545105.0420577896".
the Y of the a69r4 is "1421816.6166125853".

[create a70r4]
a70r4 is a area. "An area (70) in r4".
the printed name of the a70r4 is "Room 4".
Understand "Area 70 in Room 4" as a70r4.
the parent of the a70r4 is r4.
the X of the a70r4 is "545894.0".
the Y of the a70r4 is "1421880.0".

[create a71r4]
a71r4 is a area. "An area (71) in r4".
the printed name of the a71r4 is "Room 4".
Understand "Area 71 in Room 4" as a71r4.
the parent of the a71r4 is r4.
the X of the a71r4 is "545866.849158728".
the Y of the a71r4 is "1421860.2680626006".

[create a72r4]
a72r4 is a area. "An area (72) in r4".
the printed name of the a72r4 is "Room 4".
Understand "Area 72 in Room 4" as a72r4.
the parent of the a72r4 is r4.
the X of the a72r4 is "545065.0".
the Y of the a72r4 is "1421880.0".

[create a73r4]
a73r4 is a area. "An area (73) in r4".
the printed name of the a73r4 is "Room 4".
Understand "Area 73 in Room 4" as a73r4.
the parent of the a73r4 is r4.
the X of the a73r4 is "545096.6769635527".
the Y of the a73r4 is "1421860.7257603593".

[create a33r5]
a33r5 is a area. "An area (33) in r5".
the printed name of the a33r5 is "Room 5".
Understand "Area 33 in Room 5" as a33r5.
the parent of the a33r5 is r5.
the X of the a33r5 is "548526.8286071357".
the Y of the a33r5 is "1423742.4249786977".

[create a34r5]
a34r5 is a area. "An area (34) in r5".
the printed name of the a34r5 is "Room 5".
Understand "Area 34 in Room 5" as a34r5.
the parent of the a34r5 is r5.
the X of the a34r5 is "548549.6661241965".
the Y of the a34r5 is "1423700.8474952483".

[create a35r5]
a35r5 is a area. "An area (35) in r5".
the printed name of the a35r5 is "Room 5".
Understand "Area 35 in Room 5" as a35r5.
the parent of the a35r5 is r5.
the X of the a35r5 is "548468.4021739131".
the Y of the a35r5 is "1423763.5525401558".

[create a56r5]
a56r5 is a area. "An area (56) in r5".
the printed name of the a56r5 is "Room 5".
Understand "Area 56 in Room 5" as a56r5.
the parent of the a56r5 is r5.
the X of the a56r5 is "548565.0".
the Y of the a56r5 is "1423775.0".

[create a57r5]
a57r5 is a area. "An area (57) in r5".
the printed name of the a57r5 is "Room 5".
Understand "Area 57 in Room 5" as a57r5.
the parent of the a57r5 is r5.
the X of the a57r5 is "548532.8007246378".
the Y of the a57r5 is "1423750.169354839".

[create a84r5]
a84r5 is a area. "An area (84) in r5".
the printed name of the a84r5 is "Room 5".
Understand "Area 84 in Room 5" as a84r5.
the parent of the a84r5 is r5.
the X of the a84r5 is "548565.0".
the Y of the a84r5 is "1423675.677419355".

[create a85r5]
a85r5 is a area. "An area (85) in r5".
the printed name of the a85r5 is "Room 5".
Understand "Area 85 in Room 5" as a85r5.
the parent of the a85r5 is r5.
the X of the a85r5 is "548390.0".
the Y of the a85r5 is "1423766.0".

[create a86r5]
a86r5 is a area. "An area (86) in r5".
the printed name of the a86r5 is "Room 5".
Understand "Area 86 in Room 5" as a86r5.
the parent of the a86r5 is r5.
the X of the a86r5 is "548371.8043478262".
the Y of the a86r5 is "1423775.0".

[create d0]
d0 is a door. "A door between a21r3 and a27r4".
d0 is south of a21r3 and north of a27r4.

[create d2]
d2 is a door. "A door between a22r0 and a21r3".
d2 is northwest of a22r0 and southeast of a21r3.

[create d4]
d4 is a door. "A door between a17r0 and a18r2".
d4 is north of a17r0 and south of a18r2.

[create d6]
d6 is a door. "A door between a4r0 and a3r1".
d6 is north of a4r0 and south of a3r1.

[create d7]
d7 is a door. "A door between a5r0 and a7r1".
d7 is north of a5r0 and south of a7r1.

north of a0r0 is south of a1r0.

north of a1r0 is south of a2r0.

north of a2r0 is south of a5r0.

north of a3r1 is south of a6r1.

north of a6r1 is south of a8r1.

north of a7r1 is south of a9r1.

south of a10r0 is north of a11r0.

north of a12r0 is south of a13r0.

north of a10r0 is south of a14r0.

north of a13r0 is south of a15r0.

north of a14r0 is south of a16r0.

south of a0r0 is north of a15r0.

south of a4r0 is north of a16r0.

northeast of a19r0 is southwest of a20r0.

east of a20r0 is west of a23r0.

east of a23r0 is west of a24r0.

south of a12r0 is north of a24r0.

north of a23r0 is south of a25r0.

north of a25r0 is south of a26r0.

south of a11r0 is north of a26r0.

east of a27r4 is west of a28r4.

west of a27r4 is east of a29r4.

east of a24r0 is west of a30r0.

southwest of a17r0 is northeast of a30r0.

east of a31r3 is west of a32r3.

southeast of a33r5 is northwest of a34r5.

west of a33r5 is east of a35r5.

east of a36r1 is west of a37r1.

north of a21r3 is south of a32r3.

east of a38r1 is west of a39r1.

east of a40r1 is west of a41r1.

southwest of a36r1 is northeast of a41r1.

southeast of a39r1 is northwest of a40r1.

north of a8r1 is south of a40r1.

north of a9r1 is south of a42r1.

north of a42r1 is south of a43r1.

north of a43r1 is south of a44r1.

north of a44r1 is south of a45r1.

north of a45r1 is south of a46r1.

north of a46r1 is south of a47r1.

south of a41r1 is north of a47r1.

northeast of a48r0 is southwest of a49r0.

northwest of a50r0 is southeast of a51r0.

north of a52r0 is south of a53r0.

west of a54r0 is east of a55r0.

southwest of a56r5 is northeast of a57r5.

southwest of a58r3 is northeast of a59r3.

northeast of a60r3 is southwest of a61r3.

northeast of a62r1 is southwest of a63r1.

southeast of a64r3 is northwest of a65r3.

northwest of a66r4 is southeast of a67r4.

northeast of a68r4 is southwest of a69r4.

southwest of a70r4 is northeast of a71r4.

southeast of a72r4 is northwest of a73r4.

south of a74r0 is north of a75r0.

northwest of a76r0 is southeast of a77r0.

southeast of a78r1 is northwest of a79r1.

east of a80r1 is west of a81r1.

west of a82r1 is east of a83r1.

northwest of a18r2 is southeast of a84r2.

northwest of a84r2 is southeast of a34r5.

west of a35r5 is east of a85r5.

east of a37r1 is west of a86r1.

north of a39r1 is south of a81r1.

north of a36r1 is south of a83r1.

east of a17r0 is west of a55r0.

west of a47r1 is east of a87r1.

northeast of a33r5 is southwest of a57r5.

northeast of a32r3 is southwest of a59r3.

southwest of a31r3 is northeast of a61r3.

southwest of a38r1 is northeast of a63r1.

northwest of a31r3 is southeast of a65r3.

northwest of a38r1 is southeast of a79r1.

northwest of a29r4 is southeast of a73r4.

east of a75r0 is west of a88r0.

northeast of a77r0 is southwest of a88r0.

southwest of a19r0 is northeast of a88r0.

south of a19r0 is north of a49r0.

southeast of a30r0 is northwest of a51r0.

southwest of a22r0 is northeast of a53r0.

west of a20r0 is east of a22r0.

southeast of a28r4 is northwest of a67r4.

southwest of a29r4 is northeast of a69r4.

northeast of a28r4 is southwest of a71r4.

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
the player is in a70r4.

the orientation of the player is 6.
