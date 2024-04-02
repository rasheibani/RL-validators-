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

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "547444.0".
the Y of the a22r0 is "1211114.0".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "547475.0876068376".
the Y of the a23r0 is "1211088.1662393163".

[create a24r0]
a24r0 is a area. "An area (24) in r0".
the printed name of the a24r0 is "Room 0".
Understand "Area 24 in Room 0" as a24r0.
the parent of the a24r0 is r0.
the X of the a24r0 is "548203.0".
the Y of the a24r0 is "1211114.0".

[create a25r0]
a25r0 is a area. "An area (25) in r0".
the printed name of the a25r0 is "Room 0".
Understand "Area 25 in Room 0" as a25r0.
the parent of the a25r0 is r0.
the X of the a25r0 is "548171.9123931624".
the Y of the a25r0 is "1211088.1662393163".

[create a32r0]
a32r0 is a area. "An area (32) in r0".
the printed name of the a32r0 is "Room 0".
Understand "Area 32 in Room 0" as a32r0.
the parent of the a32r0 is r0.
the X of the a32r0 is "547725.6827814153".
the Y of the a32r0 is "1210323.9195767355".

[create a41r0]
a41r0 is a area. "An area (41) in r0".
the printed name of the a41r0 is "Room 0".
Understand "Area 41 in Room 0" as a41r0.
the parent of the a41r0 is r0.
the X of the a41r0 is "547435.0".
the Y of the a41r0 is "1211024.0".

[create a42r0]
a42r0 is a area. "An area (42) in r0".
the printed name of the a42r0 is "Room 0".
Understand "Area 42 in Room 0" as a42r0.
the parent of the a42r0 is r0.
the X of the a42r0 is "547475.7622699386".
the Y of the a42r0 is "1211042.6487730062".

[create a43r0]
a43r0 is a area. "An area (43) in r0".
the printed name of the a43r0 is "Room 0".
Understand "Area 43 in Room 0" as a43r0.
the parent of the a43r0 is r0.
the X of the a43r0 is "548212.0".
the Y of the a43r0 is "1211024.0".

[create a44r0]
a44r0 is a area. "An area (44) in r0".
the printed name of the a44r0 is "Room 0".
Understand "Area 44 in Room 0" as a44r0.
the parent of the a44r0 is r0.
the X of the a44r0 is "548173.1323529412".
the Y of the a44r0 is "1211042.838235294".

[create a69r0]
a69r0 is a area. "An area (69) in r0".
the printed name of the a69r0 is "Room 0".
Understand "Area 69 in Room 0" as a69r0.
the parent of the a69r0 is r0.
the X of the a69r0 is "547491.5405048295".
the Y of the a69r0 is "1211060.8544286496".

[create a70r0]
a70r0 is a area. "An area (70) in r0".
the printed name of the a70r0 is "Room 0".
Understand "Area 70 in Room 0" as a70r0.
the parent of the a70r0 is r0.
the X of the a70r0 is "548155.8307569793".
the Y of the a70r0 is "1211061.4707232527".

[create a76r0]
a76r0 is a area. "An area (76) in r0".
the printed name of the a76r0 is "Room 0".
Understand "Area 76 in Room 0" as a76r0.
the parent of the a76r0 is r0.
the X of the a76r0 is "547642.3050765235".
the Y of the a76r0 is "1210355.002785119".

[create a96r0]
a96r0 is a area. "An area (96) in r0".
the printed name of the a96r0 is "Room 0".
Understand "Area 96 in Room 0" as a96r0.
the parent of the a96r0 is r0.
the X of the a96r0 is "547831.4895726717".
the Y of the a96r0 is "1210972.4549042068".

[create a4r1]
a4r1 is a area. "An area (4) in r1".
the printed name of the a4r1 is "Room 1".
Understand "Area 4 in Room 1" as a4r1.
the parent of the a4r1 is r1.
the X of the a4r1 is "547293.0".
the Y of the a4r1 is "1209000.0".

[create a5r1]
a5r1 is a area. "An area (5) in r1".
the printed name of the a5r1 is "Room 1".
Understand "Area 5 in Room 1" as a5r1.
the parent of the a5r1 is r1.
the X of the a5r1 is "547250.4517163367".
the Y of the a5r1 is "1209049.425299381".

[create a63r1]
a63r1 is a area. "An area (63) in r1".
the printed name of the a63r1 is "Room 1".
Understand "Area 63 in Room 1" as a63r1.
the parent of the a63r1 is r1.
the X of the a63r1 is "546992.0".
the Y of the a63r1 is "1208972.0".

[create a64r1]
a64r1 is a area. "An area (64) in r1".
the printed name of the a64r1 is "Room 1".
Understand "Area 64 in Room 1" as a64r1.
the parent of the a64r1 is r1.
the X of the a64r1 is "547025.0613114523".
the Y of the a64r1 is "1209024.5596518875".

[create a85r1]
a85r1 is a area. "An area (85) in r1".
the printed name of the a85r1 is "Room 1".
Understand "Area 85 in Room 1" as a85r1.
the parent of the a85r1 is r1.
the X of the a85r1 is "547186.9629832634".
the Y of the a85r1 is "1209128.2779362001".

[create a87r1]
a87r1 is a area. "An area (87) in r1".
the printed name of the a87r1 is "Room 1".
Understand "Area 87 in Room 1" as a87r1.
the parent of the a87r1 is r1.
the X of the a87r1 is "547095.6294029707".
the Y of the a87r1 is "1209133.519746642".

[create a18r2]
a18r2 is a area. "An area (18) in r2".
the printed name of the a18r2 is "Room 2".
Understand "Area 18 in Room 2" as a18r2.
the parent of the a18r2 is r2.
the X of the a18r2 is "545117.0".
the Y of the a18r2 is "1211114.0".

[create a19r2]
a19r2 is a area. "An area (19) in r2".
the printed name of the a19r2 is "Room 2".
Understand "Area 19 in Room 2" as a19r2.
the parent of the a19r2 is r2.
the X of the a19r2 is "545147.9153780066".
the Y of the a19r2 is "1211088.1834621995".

[create a20r2]
a20r2 is a area. "An area (20) in r2".
the printed name of the a20r2 is "Room 2".
Understand "Area 20 in Room 2" as a20r2.
the parent of the a20r2 is r2.
the X of the a20r2 is "546897.0".
the Y of the a20r2 is "1211114.0".

[create a21r2]
a21r2 is a area. "An area (21) in r2".
the printed name of the a21r2 is "Room 2".
Understand "Area 21 in Room 2" as a21r2.
the parent of the a21r2 is r2.
the X of the a21r2 is "546866.0846219931".
the Y of the a21r2 is "1211088.1834621993".

[create a26r2]
a26r2 is a area. "An area (26) in r2".
the printed name of the a26r2 is "Room 2".
Understand "Area 26 in Room 2" as a26r2.
the parent of the a26r2 is r2.
the X of the a26r2 is "546083.0".
the Y of the a26r2 is "1211114.0".

[create a27r2]
a27r2 is a area. "An area (27) in r2".
the printed name of the a27r2 is "Room 2".
Understand "Area 27 in Room 2" as a27r2.
the parent of the a27r2 is r2.
the X of the a27r2 is "546114.9940170939".
the Y of the a27r2 is "1211105.7505982907".

[create a28r2]
a28r2 is a area. "An area (28) in r2".
the printed name of the a28r2 is "Room 2".
Understand "Area 28 in Room 2" as a28r2.
the parent of the a28r2 is r2.
the X of the a28r2 is "546631.0".
the Y of the a28r2 is "1211123.0".

[create a29r2]
a29r2 is a area. "An area (29) in r2".
the printed name of the a29r2 is "Room 2".
Understand "Area 29 in Room 2" as a29r2.
the parent of the a29r2 is r2.
the X of the a29r2 is "546600.037202381".
the Y of the a29r2 is "1211113.5".

[create a37r2]
a37r2 is a area. "An area (37) in r2".
the printed name of the a37r2 is "Room 2".
Understand "Area 37 in Room 2" as a37r2.
the parent of the a37r2 is r2.
the X of the a37r2 is "545108.0".
the Y of the a37r2 is "1211024.0".

[create a38r2]
a38r2 is a area. "An area (38) in r2".
the printed name of the a38r2 is "Room 2".
Understand "Area 38 in Room 2" as a38r2.
the parent of the a38r2 is r2.
the X of the a38r2 is "545150.06".
the Y of the a38r2 is "1211042.5189999999".

[create a39r2]
a39r2 is a area. "An area (39) in r2".
the printed name of the a39r2 is "Room 2".
Understand "Area 39 in Room 2" as a39r2.
the parent of the a39r2 is r2.
the X of the a39r2 is "546906.0".
the Y of the a39r2 is "1211024.0".

[create a40r2]
a40r2 is a area. "An area (40) in r2".
the printed name of the a40r2 is "Room 2".
Understand "Area 40 in Room 2" as a40r2.
the parent of the a40r2 is r2.
the X of the a40r2 is "546865.5982564336".
the Y of the a40r2 is "1211042.6848256432".

[create a57r2]
a57r2 is a area. "An area (57) in r2".
the printed name of the a57r2 is "Room 2".
Understand "Area 57 in Room 2" as a57r2.
the parent of the a57r2 is r2.
the X of the a57r2 is "546392.0".
the Y of the a57r2 is "1210410.0".

[create a58r2]
a58r2 is a area. "An area (58) in r2".
the printed name of the a58r2 is "Room 2".
Understand "Area 58 in Room 2" as a58r2.
the parent of the a58r2 is r2.
the X of the a58r2 is "546361.7769924196".
the Y of the a58r2 is "1210383.9929317762".

[create a59r2]
a59r2 is a area. "An area (59) in r2".
the printed name of the a59r2 is "Room 2".
Understand "Area 59 in Room 2" as a59r2.
the parent of the a59r2 is r2.
the X of the a59r2 is "546284.0".
the Y of the a59r2 is "1210422.0".

[create a60r2]
a60r2 is a area. "An area (60) in r2".
the printed name of the a60r2 is "Room 2".
Understand "Area 60 in Room 2" as a60r2.
the parent of the a60r2 is r2.
the X of the a60r2 is "546307.7621979022".
the Y of the a60r2 is "1210389.859781122".

[create a67r2]
a67r2 is a area. "An area (67) in r2".
the printed name of the a67r2 is "Room 2".
Understand "Area 67 in Room 2" as a67r2.
the parent of the a67r2 is r2.
the X of the a67r2 is "546578.4311144294".
the Y of the a67r2 is "1211017.9783480037".

[create a68r2]
a68r2 is a area. "An area (68) in r2".
the printed name of the a68r2 is "Room 2".
Understand "Area 68 in Room 2" as a68r2.
the parent of the a68r2 is r2.
the X of the a68r2 is "546849.6927655565".
the Y of the a68r2 is "1211061.1041153662".

[create a71r2]
a71r2 is a area. "An area (71) in r2".
the printed name of the a71r2 is "Room 2".
Understand "Area 71 in Room 2" as a71r2.
the parent of the a71r2 is r2.
the X of the a71r2 is "545165.0926719012".
the Y of the a71r2 is "1211059.8065726862".

[create a72r2]
a72r2 is a area. "An area (72) in r2".
the printed name of the a72r2 is "Room 2".
Understand "Area 72 in Room 2" as a72r2.
the parent of the a72r2 is r2.
the X of the a72r2 is "546121.1311216608".
the Y of the a72r2 is "1211045.3001183087".

[create a74r2]
a74r2 is a area. "An area (74) in r2".
the printed name of the a74r2 is "Room 2".
Understand "Area 74 in Room 2" as a74r2.
the parent of the a74r2 is r2.
the X of the a74r2 is "546328.858357395".
the Y of the a74r2 is "1210360.983477703".

[create a88r2]
a88r2 is a area. "An area (88) in r2".
the printed name of the a88r2 is "Room 2".
Understand "Area 88 in Room 2" as a88r2.
the parent of the a88r2 is r2.
the X of the a88r2 is "546479.7414505817".
the Y of the a88r2 is "1210955.3330843726".

[create a90r2]
a90r2 is a area. "An area (90) in r2".
the printed name of the a90r2 is "Room 2".
Understand "Area 90 in Room 2" as a90r2.
the parent of the a90r2 is r2.
the X of the a90r2 is "545526.8626086956".
the Y of the a90r2 is "1210066.0".

[create a92r2]
a92r2 is a area. "An area (92) in r2".
the printed name of the a92r2 is "Room 2".
Understand "Area 92 in Room 2" as a92r2.
the parent of the a92r2 is r2.
the X of the a92r2 is "546259.7255000002".
the Y of the a92r2 is "1210066.0".

[create a95r2]
a95r2 is a area. "An area (95) in r2".
the printed name of the a95r2 is "Room 2".
Understand "Area 95 in Room 2" as a95r2.
the parent of the a95r2 is r2.
the X of the a95r2 is "545524.8877729874".
the Y of the a95r2 is "1210933.7560030778".

[create a0r3]
a0r3 is a area. "An area (0) in r3".
the printed name of the a0r3 is "Room 3".
Understand "Area 0 in Room 3" as a0r3.
the parent of the a0r3 is r3.
the X of the a0r3 is "545083.0".
the Y of the a0r3 is "1209080.0".

[create a1r3]
a1r3 is a area. "An area (1) in r3".
the printed name of the a1r3 is "Room 3".
Understand "Area 1 in Room 3" as a1r3.
the parent of the a1r3 is r3.
the X of the a1r3 is "545114.0059757696".
the Y of the a1r3 is "1209060.657907662".

[create a8r3]
a8r3 is a area. "An area (8) in r3".
the printed name of the a8r3 is "Room 3".
Understand "Area 8 in Room 3" as a8r3.
the parent of the a8r3 is r3.
the X of the a8r3 is "546763.0".
the Y of the a8r3 is "1209488.0".

[create a9r3]
a9r3 is a area. "An area (9) in r3".
the printed name of the a9r3 is "Room 3".
Understand "Area 9 in Room 3" as a9r3.
the parent of the a9r3 is r3.
the X of the a9r3 is "546729.1511123563".
the Y of the a9r3 is "1209467.8660087995".

[create a10r3]
a10r3 is a area. "An area (10) in r3".
the printed name of the a10r3 is "Room 3".
Understand "Area 10 in Room 3" as a10r3.
the parent of the a10r3 is r3.
the X of the a10r3 is "546680.0".
the Y of the a10r3 is "1209525.0".

[create a11r3]
a11r3 is a area. "An area (11) in r3".
the printed name of the a11r3 is "Room 3".
Understand "Area 11 in Room 3" as a11r3.
the parent of the a11r3 is r3.
the X of the a11r3 is "546686.5365348754".
the Y of the a11r3 is "1209483.8657403952".

[create a14r3]
a14r3 is a area. "An area (14) in r3".
the printed name of the a14r3 is "Room 3".
Understand "Area 14 in Room 3" as a14r3.
the parent of the a14r3 is r3.
the X of the a14r3 is "546284.0".
the Y of the a14r3 is "1209710.0".

[create a15r3]
a15r3 is a area. "An area (15) in r3".
the printed name of the a15r3 is "Room 3".
Understand "Area 15 in Room 3" as a15r3.
the parent of the a15r3 is r3.
the X of the a15r3 is "546307.0755088734".
the Y of the a15r3 is "1209742.0063361116".

[create a16r3]
a16r3 is a area. "An area (16) in r3".
the printed name of the a16r3 is "Room 3".
Understand "Area 16 in Room 3" as a16r3.
the parent of the a16r3 is r3.
the X of the a16r3 is "545092.0".
the Y of the a16r3 is "1208991.0".

[create a17r3]
a17r3 is a area. "An area (17) in r3".
the printed name of the a17r3 is "Room 3".
Understand "Area 17 in Room 3" as a17r3.
the parent of the a17r3 is r3.
the X of the a17r3 is "545123.2137161085".
the Y of the a17r3 is "1209016.6339712918".

[create a45r3]
a45r3 is a area. "An area (45) in r3".
the printed name of the a45r3 is "Room 3".
Understand "Area 45 in Room 3" as a45r3.
the parent of the a45r3 is r3.
the X of the a45r3 is "546392.0".
the Y of the a45r3 is "1209725.0".

[create a46r3]
a46r3 is a area. "An area (46) in r3".
the printed name of the a46r3 is "Room 3".
Understand "Area 46 in Room 3" as a46r3.
the parent of the a46r3 is r3.
the X of the a46r3 is "546360.8077975204".
the Y of the a46r3 is "1209751.4338578528".

[create a61r3]
a61r3 is a area. "An area (61) in r3".
the printed name of the a61r3 is "Room 3".
Understand "Area 61 in Room 3" as a61r3.
the parent of the a61r3 is r3.
the X of the a61r3 is "546690.0".
the Y of the a61r3 is "1209000.0".

[create a62r3]
a62r3 is a area. "An area (62) in r3".
the printed name of the a62r3 is "Room 3".
Understand "Area 62 in Room 3" as a62r3.
the parent of the a62r3 is r3.
the X of the a62r3 is "546651.59375".
the Y of the a62r3 is "1209034.1686912226".

[create a78r3]
a78r3 is a area. "An area (78) in r3".
the printed name of the a78r3 is "Room 3".
Understand "Area 78 in Room 3" as a78r3.
the parent of the a78r3 is r3.
the X of the a78r3 is "546705.3706921167".
the Y of the a78r3 is "1209460.504141509".

[create a80r3]
a80r3 is a area. "An area (80) in r3".
the printed name of the a80r3 is "Room 3".
Understand "Area 80 in Room 3" as a80r3.
the parent of the a80r3 is r3.
the X of the a80r3 is "546328.9241359932".
the Y of the a80r3 is "1209772.8985341634".

[create a83r3]
a83r3 is a area. "An area (83) in r3".
the printed name of the a83r3 is "Room 3".
Understand "Area 83 in Room 3" as a83r3.
the parent of the a83r3 is r3.
the X of the a83r3 is "545137.3891992552".
the Y of the a83r3 is "1209040.2597765361".

[create a89r3]
a89r3 is a area. "An area (89) in r3".
the printed name of the a89r3 is "Room 3".
Understand "Area 89 in Room 3" as a89r3.
the parent of the a89r3 is r3.
the X of the a89r3 is "545524.6643713762".
the Y of the a89r3 is "1209172.6858417108".

[create a93r3]
a93r3 is a area. "An area (93) in r3".
the printed name of the a93r3 is "Room 3".
Understand "Area 93 in Room 3" as a93r3.
the parent of the a93r3 is r3.
the X of the a93r3 is "546580.1389291603".
the Y of the a93r3 is "1209113.2164822554".

[create a2r4]
a2r4 is a area. "An area (2) in r4".
the printed name of the a2r4 is "Room 4".
Understand "Area 2 in Room 4" as a2r4.
the parent of the a2r4 is r4.
the X of the a2r4 is "548355.0".
the Y of the a2r4 is "1208963.0".

[create a3r4]
a3r4 is a area. "An area (3) in r4".
the printed name of the a3r4 is "Room 4".
Understand "Area 3 in Room 4" as a3r4.
the parent of the a3r4 is r4.
the X of the a3r4 is "548391.0".
the Y of the a3r4 is "1209055.3867768594".

[create a6r4]
a6r4 is a area. "An area (6) in r4".
the printed name of the a6r4 is "Room 4".
Understand "Area 6 in Room 4" as a6r4.
the parent of the a6r4 is r4.
the X of the a6r4 is "549698.0".
the Y of the a6r4 is "1209169.0".

[create a7r4]
a7r4 is a area. "An area (7) in r4".
the printed name of the a7r4 is "Room 4".
Understand "Area 7 in Room 4" as a7r4.
the parent of the a7r4 is r4.
the X of the a7r4 is "549639.3376066436".
the Y of the a7r4 is "1209210.7846478492".

[create a12r4]
a12r4 is a area. "An area (12) in r4".
the printed name of the a12r4 is "Room 4".
Understand "Area 12 in Room 4" as a12r4.
the parent of the a12r4 is r4.
the X of the a12r4 is "547765.0".
the Y of the a12r4 is "1209513.0".

[create a13r4]
a13r4 is a area. "An area (13) in r4".
the printed name of the a13r4 is "Room 4".
Understand "Area 13 in Room 4" as a13r4.
the parent of the a13r4 is r4.
the X of the a13r4 is "547849.2818622057".
the Y of the a13r4 is "1209549.642857143".

[create a47r4]
a47r4 is a area. "An area (47) in r4".
the printed name of the a47r4 is "Room 4".
Understand "Area 47 in Room 4" as a47r4.
the parent of the a47r4 is r4.
the X of the a47r4 is "549747.0".
the Y of the a47r4 is "1209817.0".

[create a48r4]
a48r4 is a area. "An area (48) in r4".
the printed name of the a48r4 is "Room 4".
Understand "Area 48 in Room 4" as a48r4.
the parent of the a48r4 is r4.
the X of the a48r4 is "549665.1666666667".
the Y of the a48r4 is "1209845.8333333335".

[create a49r4]
a49r4 is a area. "An area (49) in r4".
the printed name of the a49r4 is "Room 4".
Understand "Area 49 in Room 4" as a49r4.
the parent of the a49r4 is r4.
the X of the a49r4 is "549142.0".
the Y of the a49r4 is "1209854.0".

[create a50r4]
a50r4 is a area. "An area (50) in r4".
the printed name of the a50r4 is "Room 4".
Understand "Area 50 in Room 4" as a50r4.
the parent of the a50r4 is r4.
the X of the a50r4 is "549184.2240682533".
the Y of the a50r4 is "1209872.207678491".

[create a51r4]
a51r4 is a area. "An area (51) in r4".
the printed name of the a51r4 is "Room 4".
Understand "Area 51 in Room 4" as a51r4.
the parent of the a51r4 is r4.
the X of the a51r4 is "549842.0".
the Y of the a51r4 is "1209845.0".

[create a52r4]
a52r4 is a area. "An area (52) in r4".
the printed name of the a52r4 is "Room 4".
Understand "Area 52 in Room 4" as a52r4.
the parent of the a52r4 is r4.
the X of the a52r4 is "549860.2704191617".
the Y of the a52r4 is "1209873.6734131735".

[create a53r4]
a53r4 is a area. "An area (53) in r4".
the printed name of the a53r4 is "Room 4".
Understand "Area 53 in Room 4" as a53r4.
the parent of the a53r4 is r4.
the X of the a53r4 is "549151.0".
the Y of the a53r4 is "1209943.0".

[create a54r4]
a54r4 is a area. "An area (54) in r4".
the printed name of the a54r4 is "Room 4".
Understand "Area 54 in Room 4" as a54r4.
the parent of the a54r4 is r4.
the X of the a54r4 is "549182.0420577896".
the Y of the a54r4 is "1209917.3833874145".

[create a55r4]
a55r4 is a area. "An area (55) in r4".
the printed name of the a55r4 is "Room 4".
Understand "Area 55 in Room 4" as a55r4.
the parent of the a55r4 is r4.
the X of the a55r4 is "549879.0".
the Y of the a55r4 is "1209943.0".

[create a56r4]
a56r4 is a area. "An area (56) in r4".
the printed name of the a56r4 is "Room 4".
Understand "Area 56 in Room 4" as a56r4.
the parent of the a56r4 is r4.
the X of the a56r4 is "549847.7862838915".
the Y of the a56r4 is "1209917.3660287082".

[create a73r4]
a73r4 is a area. "An area (73) in r4".
the printed name of the a73r4 is "Room 4".
Understand "Area 73 in Room 4" as a73r4.
the parent of the a73r4 is r4.
the X of the a73r4 is "549833.8618197278".
the Y of the a73r4 is "1209894.1585884353".

[create a79r4]
a79r4 is a area. "An area (79) in r4".
the printed name of the a79r4 is "Room 4".
Understand "Area 79 in Room 4" as a79r4.
the parent of the a79r4 is r4.
the X of the a79r4 is "547945.674298481".
the Y of the a79r4 is "1209591.579100882".

[create a81r4]
a81r4 is a area. "An area (81) in r4".
the printed name of the a81r4 is "Room 4".
Understand "Area 81 in Room 4" as a81r4.
the parent of the a81r4 is r4.
the X of the a81r4 is "549662.8477146518".
the Y of the a81r4 is "1209847.3338316958".

[create a82r4]
a82r4 is a area. "An area (82) in r4".
the printed name of the a82r4 is "Room 4".
Understand "Area 82 in Room 4" as a82r4.
the parent of the a82r4 is r4.
the X of the a82r4 is "549199.020066252".
the Y of the a82r4 is "1209889.2230761899".

[create a84r4]
a84r4 is a area. "An area (84) in r4".
the printed name of the a84r4 is "Room 4".
Understand "Area 84 in Room 4" as a84r4.
the parent of the a84r4 is r4.
the X of the a84r4 is "548408.3005528576".
the Y of the a84r4 is "1209098.8583478578".

[create a86r4]
a86r4 is a area. "An area (86) in r4".
the printed name of the a86r4 is "Room 4".
Understand "Area 86 in Room 4" as a86r4.
the parent of the a86r4 is r4.
the X of the a86r4 is "549555.6979495726".
the Y of the a86r4 is "1209266.6546164062".

[create a91r4]
a91r4 is a area. "An area (91) in r4".
the printed name of the a91r4 is "Room 4".
Understand "Area 91 in Room 4" as a91r4.
the parent of the a91r4 is r4.
the X of the a91r4 is "549553.2063772487".
the Y of the a91r4 is "1209768.2701607419".

[create a30r5]
a30r5 is a area. "An area (30) in r5".
the printed name of the a30r5 is "Room 5".
Understand "Area 30 in Room 5" as a30r5.
the parent of the a30r5 is r5.
the X of the a30r5 is "547765.0".
the Y of the a30r5 is "1210327.0".

[create a31r5]
a31r5 is a area. "An area (31) in r5".
the printed name of the a31r5 is "Room 5".
Understand "Area 31 in Room 5" as a31r5.
the parent of the a31r5 is r5.
the X of the a31r5 is "547914.0294117648".
the Y of the a31r5 is "1210289.375".

[create a33r5]
a33r5 is a area. "An area (33) in r5".
the printed name of the a33r5 is "Room 5".
Understand "Area 33 in Room 5" as a33r5.
the parent of the a33r5 is r5.
the X of the a33r5 is "549575.0".
the Y of the a33r5 is "1210588.0".

[create a34r5]
a34r5 is a area. "An area (34) in r5".
the printed name of the a34r5 is "Room 5".
Understand "Area 34 in Room 5" as a34r5.
the parent of the a34r5 is r5.
the X of the a34r5 is "549583.3098646166".
the Y of the a34r5 is "1210627.908152513".

[create a35r5]
a35r5 is a area. "An area (35) in r5".
the printed name of the a35r5 is "Room 5".
Understand "Area 35 in Room 5" as a35r5.
the parent of the a35r5 is r5.
the X of the a35r5 is "549661.0".
the Y of the a35r5 is "1210625.0".

[create a36r5]
a36r5 is a area. "An area (36) in r5".
the printed name of the a36r5 is "Room 5".
Understand "Area 36 in Room 5" as a36r5.
the parent of the a36r5 is r5.
the X of the a36r5 is "549622.9614546176".
the Y of the a36r5 is "1210654.1909433214".

[create a65r5]
a65r5 is a area. "An area (65) in r5".
the printed name of the a65r5 is "Room 5".
Understand "Area 65 in Room 5" as a65r5.
the parent of the a65r5 is r5.
the X of the a65r5 is "549624.0".
the Y of the a65r5 is "1211104.0".

[create a66r5]
a66r5 is a area. "An area (66) in r5".
the printed name of the a66r5 is "Room 5".
Understand "Area 66 in Room 5" as a66r5.
the parent of the a66r5 is r5.
the X of the a66r5 is "549588.7727272727".
the Y of the a66r5 is "1211081.0".

[create a75r5]
a75r5 is a area. "An area (75) in r5".
the printed name of the a75r5 is "Room 5".
Understand "Area 75 in Room 5" as a75r5.
the parent of the a75r5 is r5.
the X of the a75r5 is "547924.9032078378".
the Y of the a75r5 is "1210286.847362809".

[create a77r5]
a77r5 is a area. "An area (77) in r5".
the printed name of the a77r5 is "Room 5".
Understand "Area 77 in Room 5" as a77r5.
the parent of the a77r5 is r5.
the X of the a77r5 is "549607.1150421607".
the Y of the a77r5 is "1210659.7027389586".

[create a94r5]
a94r5 is a area. "An area (94) in r5".
the printed name of the a94r5 is "Room 5".
Understand "Area 94 in Room 5" as a94r5.
the parent of the a94r5 is r5.
the X of the a94r5 is "549533.8718497385".
the Y of the a94r5 is "1210995.9455539705".

[create d0]
d0 is a door. "A door between a76r0 and a85r1".
d0 is south of a76r0 and north of a85r1.

[create d2]
d2 is a door. "A door between a87r1 and a88r2".
d2 is north of a87r1 and south of a88r2.

[create d4]
d4 is a door. "A door between a90r2 and a89r3".
d4 is south of a90r2 and north of a89r3.

[create d5]
d5 is a door. "A door between a92r2 and a80r3".
d5 is south of a92r2 and north of a80r3.

[create d6]
d6 is a door. "A door between a79r4 and a75r5".
d6 is north of a79r4 and south of a75r5.

southeast of a0r3 is northwest of a1r3.

north of a2r4 is south of a3r4.

northwest of a4r1 is southeast of a5r1.

northwest of a6r4 is southeast of a7r4.

southwest of a8r3 is northeast of a9r3.

south of a10r3 is north of a11r3.

northeast of a12r4 is southwest of a13r4.

northeast of a14r3 is southwest of a15r3.

northeast of a16r3 is southwest of a17r3.

southeast of a18r2 is northwest of a19r2.

southwest of a20r2 is northeast of a21r2.

southeast of a22r0 is northwest of a23r0.

southwest of a24r0 is northeast of a25r0.

east of a26r2 is west of a27r2.

west of a28r2 is east of a29r2.

east of a30r5 is west of a31r5.

north of a33r5 is south of a34r5.

northwest of a35r5 is southeast of a36r5.

northeast of a37r2 is southwest of a38r2.

northwest of a39r2 is southeast of a40r2.

northeast of a41r0 is southwest of a42r0.

northwest of a43r0 is southeast of a44r0.

northwest of a45r3 is southeast of a46r3.

west of a47r4 is east of a48r4.

northeast of a49r4 is southwest of a50r4.

northeast of a51r4 is southwest of a52r4.

southeast of a53r4 is northwest of a54r4.

southwest of a55r4 is northeast of a56r4.

southwest of a57r2 is northeast of a58r2.

southeast of a59r2 is northwest of a60r2.

northwest of a61r3 is southeast of a62r3.

northeast of a63r1 is southwest of a64r1.

southwest of a65r5 is northeast of a66r5.

south of a29r2 is north of a67r2.

northwest of a40r2 is southeast of a68r2.

northeast of a42r0 is southwest of a69r0.

northwest of a44r0 is southeast of a70r0.

southeast of a19r2 is northwest of a71r2.

southwest of a21r2 is northeast of a68r2.

southeast of a23r0 is northwest of a69r0.

southwest of a25r0 is northeast of a70r0.

south of a27r2 is north of a72r2.

southwest of a56r4 is northeast of a73r4.

southwest of a58r2 is northeast of a74r2.

southeast of a60r2 is northwest of a74r2.

east of a31r5 is west of a75r5.

west of a32r0 is east of a76r0.

northeast of a34r5 is southwest of a77r5.

west of a36r5 is east of a77r5.

northeast of a38r2 is southwest of a71r2.

southeast of a11r3 is northwest of a78r3.

northeast of a13r4 is southwest of a79r4.

northeast of a15r3 is southwest of a80r3.

northwest of a46r3 is southeast of a80r3.

northwest of a48r4 is southeast of a81r4.

northeast of a50r4 is southwest of a82r4.

northwest of a52r4 is southeast of a73r4.

southeast of a54r4 is northwest of a82r4.

northeast of a17r3 is southwest of a83r3.

southeast of a1r3 is northwest of a83r3.

north of a3r4 is south of a84r4.

northwest of a5r1 is southeast of a85r1.

northwest of a7r4 is southeast of a86r4.

west of a9r3 is east of a78r3.

northeast of a64r1 is southwest of a87r1.

north of a86r4 is south of a91r4.

east of a82r4 is west of a91r4.

southwest of a81r4 is northeast of a91r4.

east of a83r3 is west of a89r3.

east of a84r4 is west of a86r4.

east of a89r3 is west of a93r3.

northwest of a62r3 is southeast of a93r3.

south of a78r3 is north of a93r3.

southeast of a79r4 is northwest of a84r4.

west of a85r1 is east of a87r1.

southwest of a66r5 is northeast of a94r5.

west of a72r2 is east of a95r2.

east of a71r2 is west of a95r2.

east of a72r2 is west of a88r2.

east of a69r0 is west of a96r0.

west of a70r0 is east of a96r0.

east of a67r2 is west of a68r2.

southwest of a67r2 is northeast of a88r2.

west of a73r4 is east of a81r4.

east of a90r2 is west of a92r2.

south of a74r2 is north of a92r2.

north of a90r2 is south of a95r2.

north of a76r0 is south of a96r0.

northeast of a75r5 is southwest of a94r5.

north of a77r5 is south of a94r5.

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
the player is in a25r0.

the orientation of the player is 6.
