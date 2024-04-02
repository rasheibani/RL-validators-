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

[create a20r0]
a20r0 is a area. "An area (20) in r0".
the printed name of the a20r0 is "Room 0".
Understand "Area 20 in Room 0" as a20r0.
the parent of the a20r0 is r0.
the X of the a20r0 is "546869.5933720467".
the Y of the a20r0 is "1425637.7685901083".

[create a21r0]
a21r0 is a area. "An area (21) in r0".
the printed name of the a21r0 is "Room 0".
Understand "Area 21 in Room 0" as a21r0.
the parent of the a21r0 is r0.
the X of the a21r0 is "546921.9385484892".
the Y of the a21r0 is "1425742.1180594505".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "546892.9405397377".
the Y of the a22r0 is "1425820.8199684587".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "546783.9912057456".
the Y of the a23r0 is "1425164.2196130906".

[create a24r0]
a24r0 is a area. "An area (24) in r0".
the printed name of the a24r0 is "Room 0".
Understand "Area 24 in Room 0" as a24r0.
the parent of the a24r0 is r0.
the X of the a24r0 is "546817.5483636886".
the Y of the a24r0 is "1425166.9119265778".

[create a25r0]
a25r0 is a area. "An area (25) in r0".
the printed name of the a25r0 is "Room 0".
Understand "Area 25 in Room 0" as a25r0.
the parent of the a25r0 is r0.
the X of the a25r0 is "547187.4608331178".
the Y of the a25r0 is "1425569.0977292012".

[create a26r0]
a26r0 is a area. "An area (26) in r0".
the printed name of the a26r0 is "Room 0".
Understand "Area 26 in Room 0" as a26r0.
the parent of the a26r0 is r0.
the X of the a26r0 is "546622.686594203".
the Y of the a26r0 is "1425289.6751110335".

[create a74r0]
a74r0 is a area. "An area (74) in r0".
the printed name of the a74r0 is "Room 0".
Understand "Area 74 in Room 0" as a74r0.
the parent of the a74r0 is r0.
the X of the a74r0 is "546765.0".
the Y of the a74r0 is "1425053.0".

[create a75r0]
a75r0 is a area. "An area (75) in r0".
the printed name of the a75r0 is "Room 0".
Understand "Area 75 in Room 0" as a75r0.
the parent of the a75r0 is r0.
the X of the a75r0 is "546751.0".
the Y of the a75r0 is "1425114.4050772628".

[create a76r0]
a76r0 is a area. "An area (76) in r0".
the printed name of the a76r0 is "Room 0".
Understand "Area 76 in Room 0" as a76r0.
the parent of the a76r0 is r0.
the X of the a76r0 is "546851.0".
the Y of the a76r0 is "1425053.0".

[create a77r0]
a77r0 is a area. "An area (77) in r0".
the printed name of the a77r0 is "Room 0".
Understand "Area 77 in Room 0" as a77r0.
the parent of the a77r0 is r0.
the X of the a77r0 is "546829.5".
the Y of the a77r0 is "1425138.00390625".

[create a82r0]
a82r0 is a area. "An area (82) in r0".
the printed name of the a82r0 is "Room 0".
Understand "Area 82 in Room 0" as a82r0.
the parent of the a82r0 is r0.
the X of the a82r0 is "546624.0".
the Y of the a82r0 is "1425339.0".

[create a83r0]
a83r0 is a area. "An area (83) in r0".
the printed name of the a83r0 is "Room 0".
Understand "Area 83 in Room 0" as a83r0.
the parent of the a83r0 is r0.
the X of the a83r0 is "546846.0".
the Y of the a83r0 is "1425572.0".

[create a9r1]
a9r1 is a area. "An area (9) in r1".
the printed name of the a9r1 is "Room 1".
Understand "Area 9 in Room 1" as a9r1.
the parent of the a9r1 is r1.
the X of the a9r1 is "545503.89814202".
the Y of the a9r1 is "1427556.089074554".

[create a10r1]
a10r1 is a area. "An area (10) in r1".
the printed name of the a10r1 is "Room 1".
Understand "Area 10 in Room 1" as a10r1.
the parent of the a10r1 is r1.
the X of the a10r1 is "545554.0365723589".
the Y of the a10r1 is "1427432.6014533546".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "545168.092671901".
the Y of the a14r1 is "1427659.8065726862".

[create a42r1]
a42r1 is a area. "An area (42) in r1".
the printed name of the a42r1 is "Room 1".
Understand "Area 42 in Room 1" as a42r1.
the parent of the a42r1 is r1.
the X of the a42r1 is "545111.0".
the Y of the a42r1 is "1427624.0".

[create a43r1]
a43r1 is a area. "An area (43) in r1".
the printed name of the a43r1 is "Room 1".
Understand "Area 43 in Room 1" as a43r1.
the parent of the a43r1 is r1.
the X of the a43r1 is "545153.0599999999".
the Y of the a43r1 is "1427642.5189999999".

[create a56r1]
a56r1 is a area. "An area (56) in r1".
the printed name of the a56r1 is "Room 1".
Understand "Area 56 in Room 1" as a56r1.
the parent of the a56r1 is r1.
the X of the a56r1 is "545120.0".
the Y of the a56r1 is "1427714.0".

[create a57r1]
a57r1 is a area. "An area (57) in r1".
the printed name of the a57r1 is "Room 1".
Understand "Area 57 in Room 1" as a57r1.
the parent of the a57r1 is r1.
the X of the a57r1 is "545150.9153780069".
the Y of the a57r1 is "1427688.1834621993".

[create a72r1]
a72r1 is a area. "An area (72) in r1".
the printed name of the a72r1 is "Room 1".
Understand "Area 72 in Room 1" as a72r1.
the parent of the a72r1 is r1.
the X of the a72r1 is "545578.0".
the Y of the a72r1 is "1427704.0".

[create a73r1]
a73r1 is a area. "An area (73) in r1".
the printed name of the a73r1 is "Room 1".
Understand "Area 73 in Room 1" as a73r1.
the parent of the a73r1 is r1.
the X of the a73r1 is "545544.875".
the Y of the a73r1 is "1427630.0006598178".

[create a18r2]
a18r2 is a area. "An area (18) in r2".
the printed name of the a18r2 is "Room 2".
Understand "Area 18 in Room 2" as a18r2.
the parent of the a18r2 is r2.
the X of the a18r2 is "545119.3147627444".
the Y of the a18r2 is "1425640.2898139763".

[create a19r2]
a19r2 is a area. "An area (19) in r2".
the printed name of the a19r2 is "Room 2".
Understand "Area 19 in Room 2" as a19r2.
the parent of the a19r2 is r2.
the X of the a19r2 is "545483.4387151967".
the Y of the a19r2 is "1425730.6903873074".

[create a28r2]
a28r2 is a area. "An area (28) in r2".
the printed name of the a28r2 is "Room 2".
Understand "Area 28 in Room 2" as a28r2.
the parent of the a28r2 is r2.
the X of the a28r2 is "545839.7586021506".
the Y of the a28r2 is "1425640.0134408602".

[create a60r2]
a60r2 is a area. "An area (60) in r2".
the printed name of the a60r2 is "Room 2".
Understand "Area 60 in Room 2" as a60r2.
the parent of the a60r2 is r2.
the X of the a60r2 is "545894.0".
the Y of the a60r2 is "1425680.0".

[create a61r2]
a61r2 is a area. "An area (61) in r2".
the printed name of the a61r2 is "Room 2".
Understand "Area 61 in Room 2" as a61r2.
the parent of the a61r2 is r2.
the X of the a61r2 is "545866.849158728".
the Y of the a61r2 is "1425660.2680626004".

[create a64r2]
a64r2 is a area. "An area (64) in r2".
the printed name of the a64r2 is "Room 2".
Understand "Area 64 in Room 2" as a64r2.
the parent of the a64r2 is r2.
the X of the a64r2 is "545065.0".
the Y of the a64r2 is "1425680.0".

[create a65r2]
a65r2 is a area. "An area (65) in r2".
the printed name of the a65r2 is "Room 2".
Understand "Area 65 in Room 2" as a65r2.
the parent of the a65r2 is r2.
the X of the a65r2 is "545096.6769635527".
the Y of the a65r2 is "1425660.7257603593".

[create a78r2]
a78r2 is a area. "An area (78) in r2".
the printed name of the a78r2 is "Room 2".
Understand "Area 78 in Room 2" as a78r2.
the parent of the a78r2 is r2.
the X of the a78r2 is "545885.0".
the Y of the a78r2 is "1425591.0".

[create a79r2]
a79r2 is a area. "An area (79) in r2".
the printed name of the a79r2 is "Room 2".
Understand "Area 79 in Room 2" as a79r2.
the parent of the a79r2 is r2.
the X of the a79r2 is "545853.7862838916".
the Y of the a79r2 is "1425616.633971292".

[create a80r2]
a80r2 is a area. "An area (80) in r2".
the printed name of the a80r2 is "Room 2".
Understand "Area 80 in Room 2" as a80r2.
the parent of the a80r2 is r2.
the X of the a80r2 is "545074.0".
the Y of the a80r2 is "1425591.0".

[create a81r2]
a81r2 is a area. "An area (81) in r2".
the printed name of the a81r2 is "Room 2".
Understand "Area 81 in Room 2" as a81r2.
the parent of the a81r2 is r2.
the X of the a81r2 is "545105.0420577895".
the Y of the a81r2 is "1425616.6166125853".

[create a0r3]
a0r3 is a area. "An area (0) in r3".
the printed name of the a0r3 is "Room 3".
Understand "Area 0 in Room 3" as a0r3.
the parent of the a0r3 is r3.
the X of the a0r3 is "548451.707266629".
the Y of the a0r3 is "1427661.4777749958".

[create a1r3]
a1r3 is a area. "An area (1) in r3".
the printed name of the a1r3 is "Room 3".
Understand "Area 1 in Room 3" as a1r3.
the parent of the a1r3 is r3.
the X of the a1r3 is "548795.3793965107".
the Y of the a1r3 is "1427574.0644405335".

[create a2r3]
a2r3 is a area. "An area (2) in r3".
the printed name of the a2r3 is "Room 3".
Understand "Area 2 in Room 3" as a2r3.
the parent of the a2r3 is r3.
the X of the a2r3 is "549116.305100284".
the Y of the a2r3 is "1427661.49835354".

[create a3r3]
a3r3 is a area. "An area (3) in r3".
the printed name of the a3r3 is "Room 3".
Understand "Area 3 in Room 3" as a3r3.
the parent of the a3r3 is r3.
the X of the a3r3 is "546900.0187317727".
the Y of the a3r3 is "1427626.7071587027".

[create a4r3]
a4r3 is a area. "An area (4) in r3".
the printed name of the a4r3 is "Room 3".
Understand "Area 4 in Room 3" as a4r3.
the parent of the a4r3 is r3.
the X of the a4r3 is "546967.4743406621".
the Y of the a4r3 is "1427578.1322161572".

[create a5r3]
a5r3 is a area. "An area (5) in r3".
the printed name of the a5r3 is "Room 3".
Understand "Area 5 in Room 3" as a5r3.
the parent of the a5r3 is r3.
the X of the a5r3 is "547626.6687285053".
the Y of the a5r3 is "1427582.463758903".

[create a6r3]
a6r3 is a area. "An area (6) in r3".
the printed name of the a6r3 is "Room 3".
Understand "Area 6 in Room 3" as a6r3.
the parent of the a6r3 is r3.
the X of the a6r3 is "547973.915169287".
the Y of the a6r3 is "1427661.610847683".

[create a7r3]
a7r3 is a area. "An area (7) in r3".
the printed name of the a7r3 is "Room 3".
Understand "Area 7 in Room 3" as a7r3.
the parent of the a7r3 is r3.
the X of the a7r3 is "548160.1730804583".
the Y of the a7r3 is "1426459.7005024815".

[create a8r3]
a8r3 is a area. "An area (8) in r3".
the printed name of the a8r3 is "Room 3".
Understand "Area 8 in Room 3" as a8r3.
the parent of the a8r3 is r3.
the X of the a8r3 is "547590.6312753351".
the Y of the a8r3 is "1427543.6541939508".

[create a11r3]
a11r3 is a area. "An area (11) in r3".
the printed name of the a11r3 is "Room 3".
Understand "Area 11 in Room 3" as a11r3.
the parent of the a11r3 is r3.
the X of the a11r3 is "546967.4999999999".
the Y of the a11r3 is "1427481.7578268873".

[create a12r3]
a12r3 is a area. "An area (12) in r3".
the printed name of the a12r3 is "Room 3".
Understand "Area 12 in Room 3" as a12r3.
the parent of the a12r3 is r3.
the X of the a12r3 is "547223.197879546".
the Y of the a12r3 is "1427583.0635828872".

[create a13r3]
a13r3 is a area. "An area (13) in r3".
the printed name of the a13r3 is "Room 3".
Understand "Area 13 in Room 3" as a13r3.
the parent of the a13r3 is r3.
the X of the a13r3 is "547286.7607018162".
the Y of the a13r3 is "1427567.363140249".

[create a15r3]
a15r3 is a area. "An area (15) in r3".
the printed name of the a15r3 is "Room 3".
Understand "Area 15 in Room 3" as a15r3.
the parent of the a15r3 is r3.
the X of the a15r3 is "546612.4446009533".
the Y of the a15r3 is "1427664.1811858912".

[create a27r3]
a27r3 is a area. "An area (27) in r3".
the printed name of the a27r3 is "Room 3".
Understand "Area 27 in Room 3" as a27r3.
the parent of the a27r3 is r3.
the X of the a27r3 is "547261.1032979904".
the Y of the a27r3 is "1427546.0339928542".

[create a30r3]
a30r3 is a area. "An area (30) in r3".
the printed name of the a30r3 is "Room 3".
Understand "Area 30 in Room 3" as a30r3.
the parent of the a30r3 is r3.
the X of the a30r3 is "546567.0".
the Y of the a30r3 is "1427714.0".

[create a31r3]
a31r3 is a area. "An area (31) in r3".
the printed name of the a31r3 is "Room 3".
Understand "Area 31 in Room 3" as a31r3.
the parent of the a31r3 is r3.
the X of the a31r3 is "546597.9153780069".
the Y of the a31r3 is "1427688.1834621993".

[create a32r3]
a32r3 is a area. "An area (32) in r3".
the printed name of the a32r3 is "Room 3".
Understand "Area 32 in Room 3" as a32r3.
the parent of the a32r3 is r3.
the X of the a32r3 is "548021.0".
the Y of the a32r3 is "1427714.0".

[create a33r3]
a33r3 is a area. "An area (33) in r3".
the printed name of the a33r3 is "Room 3".
Understand "Area 33 in Room 3" as a33r3.
the parent of the a33r3 is r3.
the X of the a33r3 is "547989.9123931623".
the Y of the a33r3 is "1427688.1662393163".

[create a34r3]
a34r3 is a area. "An area (34) in r3".
the printed name of the a34r3 is "Room 3".
Understand "Area 34 in Room 3" as a34r3.
the parent of the a34r3 is r3.
the X of the a34r3 is "547671.0".
the Y of the a34r3 is "1427714.0".

[create a35r3]
a35r3 is a area. "An area (35) in r3".
the printed name of the a35r3 is "Room 3".
Understand "Area 35 in Room 3" as a35r3.
the parent of the a35r3 is r3.
the X of the a35r3 is "547639.1923615087".
the Y of the a35r3 is "1427705.4040895505".

[create a36r3]
a36r3 is a area. "An area (36) in r3".
the printed name of the a36r3 is "Room 3".
Understand "Area 36 in Room 3" as a36r3.
the parent of the a36r3 is r3.
the X of the a36r3 is "546860.0".
the Y of the a36r3 is "1427714.0".

[create a37r3]
a37r3 is a area. "An area (37) in r3".
the printed name of the a37r3 is "Room 3".
Understand "Area 37 in Room 3" as a37r3.
the parent of the a37r3 is r3.
the X of the a37r3 is "546891.994017094".
the Y of the a37r3 is "1427705.7505982907".

[create a38r3]
a38r3 is a area. "An area (38) in r3".
the printed name of the a38r3 is "Room 3".
Understand "Area 38 in Room 3" as a38r3.
the parent of the a38r3 is r3.
the X of the a38r3 is "547255.3023255814".
the Y of the a38r3 is "1427710.046511628".

[create a39r3]
a39r3 is a area. "An area (39) in r3".
the printed name of the a39r3 is "Room 3".
Understand "Area 39 in Room 3" as a39r3.
the parent of the a39r3 is r3.
the X of the a39r3 is "547216.8500421739".
the Y of the a39r3 is "1427706.2116279071".

[create a40r3]
a40r3 is a area. "An area (40) in r3".
the printed name of the a40r3 is "Room 3".
Understand "Area 40 in Room 3" as a40r3.
the parent of the a40r3 is r3.
the X of the a40r3 is "547332.0".
the Y of the a40r3 is "1427714.0".

[create a41r3]
a41r3 is a area. "An area (41) in r3".
the printed name of the a41r3 is "Room 3".
Understand "Area 41 in Room 3" as a41r3.
the parent of the a41r3 is r3.
the X of the a41r3 is "547293.95282219".
the Y of the a41r3 is "1427706.1710634693".

[create a44r3]
a44r3 is a area. "An area (44) in r3".
the printed name of the a44r3 is "Room 3".
Understand "Area 44 in Room 3" as a44r3.
the parent of the a44r3 is r3.
the X of the a44r3 is "548030.0".
the Y of the a44r3 is "1427624.0".

[create a45r3]
a45r3 is a area. "An area (45) in r3".
the printed name of the a45r3 is "Room 3".
Understand "Area 45 in Room 3" as a45r3.
the parent of the a45r3 is r3.
the X of the a45r3 is "547990.9448424068".
the Y of the a45r3 is "1427642.8194842408".

[create a46r3]
a46r3 is a area. "An area (46) in r3".
the printed name of the a46r3 is "Room 3".
Understand "Area 46 in Room 3" as a46r3.
the parent of the a46r3 is r3.
the X of the a46r3 is "548395.0".
the Y of the a46r3 is "1427624.0".

[create a47r3]
a47r3 is a area. "An area (47) in r3".
the printed name of the a47r3 is "Room 3".
Understand "Area 47 in Room 3" as a47r3.
the parent of the a47r3 is r3.
the X of the a47r3 is "548433.7029286583".
the Y of the a47r3 is "1427642.477452371".

[create a48r3]
a48r3 is a area. "An area (48) in r3".
the printed name of the a48r3 is "Room 3".
Understand "Area 48 in Room 3" as a48r3.
the parent of the a48r3 is r3.
the X of the a48r3 is "549173.0".
the Y of the a48r3 is "1427624.0".

[create a49r3]
a49r3 is a area. "An area (49) in r3".
the printed name of the a49r3 is "Room 3".
Understand "Area 49 in Room 3" as a49r3.
the parent of the a49r3 is r3.
the X of the a49r3 is "549133.9982014389".
the Y of the a49r3 is "1427642.4442446043".

[create a50r3]
a50r3 is a area. "An area (50) in r3".
the printed name of the a50r3 is "Room 3".
Understand "Area 50 in Room 3" as a50r3.
the parent of the a50r3 is r3.
the X of the a50r3 is "546558.0".
the Y of the a50r3 is "1427624.0".

[create a51r3]
a51r3 is a area. "An area (51) in r3".
the printed name of the a51r3 is "Room 3".
Understand "Area 51 in Room 3" as a51r3.
the parent of the a51r3 is r3.
the X of the a51r3 is "546585.2059471365".
the Y of the a51r3 is "1427644.0044052864".

[create a52r3]
a52r3 is a area. "An area (52) in r3".
the printed name of the a52r3 is "Room 3".
Understand "Area 52 in Room 3" as a52r3.
the parent of the a52r3 is r3.
the X of the a52r3 is "548405.0".
the Y of the a52r3 is "1427714.0".

[create a53r3]
a53r3 is a area. "An area (53) in r3".
the printed name of the a53r3 is "Room 3".
Understand "Area 53 in Room 3" as a53r3.
the parent of the a53r3 is r3.
the X of the a53r3 is "548435.8949627937".
the Y of the a53r3 is "1427687.7894485786".

[create a54r3]
a54r3 is a area. "An area (54) in r3".
the printed name of the a54r3 is "Room 3".
Understand "Area 54 in Room 3" as a54r3.
the parent of the a54r3 is r3.
the X of the a54r3 is "549163.0".
the Y of the a54r3 is "1427714.0".

[create a55r3]
a55r3 is a area. "An area (55) in r3".
the printed name of the a55r3 is "Room 3".
Understand "Area 55 in Room 3" as a55r3.
the parent of the a55r3 is r3.
the X of the a55r3 is "549132.1050372064".
the Y of the a55r3 is "1427687.7894485786".

[create a70r3]
a70r3 is a area. "An area (70) in r3".
the printed name of the a70r3 is "Room 3".
Understand "Area 70 in Room 3" as a70r3.
the parent of the a70r3 is r3.
the X of the a70r3 is "547089.0".
the Y of the a70r3 is "1427492.0".

[create a71r3]
a71r3 is a area. "An area (71) in r3".
the printed name of the a71r3 is "Room 3".
Understand "Area 71 in Room 3" as a71r3.
the parent of the a71r3 is r3.
the X of the a71r3 is "547082.0480127649".
the Y of the a71r3 is "1427462.1391064695".

[create a16r4]
a16r4 is a area. "An area (16) in r4".
the printed name of the a16r4 is "Room 4".
Understand "Area 16 in Room 4" as a16r4.
the parent of the a16r4 is r4.
the X of the a16r4 is "547783.3945333334".
the Y of the a16r4 is "1425640.2686666667".

[create a17r4]
a17r4 is a area. "An area (17) in r4".
the printed name of the a17r4 is "Room 4".
Understand "Area 17 in Room 4" as a17r4.
the parent of the a17r4 is r4.
the X of the a17r4 is "548165.0".
the Y of the a17r4 is "1425768.5778301887".

[create a29r4]
a29r4 is a area. "An area (29) in r4".
the printed name of the a29r4 is "Room 4".
Understand "Area 29 in Room 4" as a29r4.
the parent of the a29r4 is r4.
the X of the a29r4 is "548531.7290448344".
the Y of the a29r4 is "1425645.062703054".

[create a58r4]
a58r4 is a area. "An area (58) in r4".
the printed name of the a58r4 is "Room 4".
Understand "Area 58 in Room 4" as a58r4.
the parent of the a58r4 is r4.
the X of the a58r4 is "547738.0".
the Y of the a58r4 is "1425591.0".

[create a59r4]
a59r4 is a area. "An area (59) in r4".
the printed name of the a59r4 is "Room 4".
Understand "Area 59 in Room 4" as a59r4.
the parent of the a59r4 is r4.
the X of the a59r4 is "547769.2137161084".
the Y of the a59r4 is "1425616.633971292".

[create a62r4]
a62r4 is a area. "An area (62) in r4".
the printed name of the a62r4 is "Room 4".
Understand "Area 62 in Room 4" as a62r4.
the parent of the a62r4 is r4.
the X of the a62r4 is "547729.0".
the Y of the a62r4 is "1425680.0".

[create a63r4]
a63r4 is a area. "An area (63) in r4".
the printed name of the a63r4 is "Room 4".
Understand "Area 63 in Room 4" as a63r4.
the parent of the a63r4 is r4.
the X of the a63r4 is "547759.8730024241".
the Y of the a63r4 is "1425660.6444609193".

[create a66r4]
a66r4 is a area. "An area (66) in r4".
the printed name of the a66r4 is "Room 4".
Understand "Area 66 in Room 4" as a66r4.
the parent of the a66r4 is r4.
the X of the a66r4 is "548580.0".
the Y of the a66r4 is "1425591.0".

[create a67r4]
a67r4 is a area. "An area (67) in r4".
the printed name of the a67r4 is "Room 4".
Understand "Area 67 in Room 4" as a67r4.
the parent of the a67r4 is r4.
the X of the a67r4 is "548548.7862838916".
the Y of the a67r4 is "1425616.633971292".

[create a68r4]
a68r4 is a area. "An area (68) in r4".
the printed name of the a68r4 is "Room 4".
Understand "Area 68 in Room 4" as a68r4.
the parent of the a68r4 is r4.
the X of the a68r4 is "548589.0".
the Y of the a68r4 is "1425680.0".

[create a69r4]
a69r4 is a area. "An area (69) in r4".
the printed name of the a69r4 is "Room 4".
Understand "Area 69 in Room 4" as a69r4.
the parent of the a69r4 is r4.
the X of the a69r4 is "548545.9383292026".
the Y of the a69r4 is "1425661.8770228897".

[create d0]
d0 is a door. "A door between a22r0 and a11r3".
d0 is north of a22r0 and south of a11r3.

[create d1]
d1 is a door. "A door between a25r0 and a27r3".
d1 is north of a25r0 and south of a27r3.

[create d2]
d2 is a door. "A door between a10r1 and a19r2".
d2 is south of a10r1 and north of a19r2.

[create d4]
d4 is a door. "A door between a7r3 and a17r4".
d4 is south of a7r3 and north of a17r4.

[create d6]
d6 is a door. "A door between a22r0 and a10r1".
d6 is northwest of a22r0 and southeast of a10r1.

east of a0r3 is west of a1r3.

east of a1r3 is west of a2r3.

southeast of a3r3 is northwest of a4r3.

east of a5r3 is west of a6r3.

southwest of a1r3 is northeast of a7r3.

northwest of a7r3 is southeast of a8r3.

south of a9r1 is north of a10r1.

south of a4r3 is north of a11r3.

east of a4r3 is west of a12r3.

west of a8r3 is east of a13r3.

west of a9r1 is east of a14r1.

west of a3r3 is east of a15r3.

east of a16r4 is west of a17r4.

east of a18r2 is west of a19r2.

northeast of a20r0 is southwest of a21r0.

north of a21r0 is south of a22r0.

east of a23r0 is west of a24r0.

northeast of a24r0 is southwest of a25r0.

northwest of a23r0 is southeast of a26r0.

southeast of a21r0 is northwest of a25r0.

east of a19r2 is west of a28r2.

east of a17r4 is west of a29r4.

southeast of a30r3 is northwest of a31r3.

southwest of a32r3 is northeast of a33r3.

west of a34r3 is east of a35r3.

east of a36r3 is west of a37r3.

west of a38r3 is east of a39r3.

west of a40r3 is east of a41r3.

northeast of a42r1 is southwest of a43r1.

northwest of a44r3 is southeast of a45r3.

northeast of a46r3 is southwest of a47r3.

northwest of a48r3 is southeast of a49r3.

northeast of a50r3 is southwest of a51r3.

southeast of a52r3 is northwest of a53r3.

southwest of a54r3 is northeast of a55r3.

southeast of a56r1 is northwest of a57r1.

northeast of a58r4 is southwest of a59r4.

southwest of a60r2 is northeast of a61r2.

southeast of a62r4 is northwest of a63r4.

southeast of a64r2 is northwest of a65r2.

northwest of a66r4 is southeast of a67r4.

southwest of a68r4 is northeast of a69r4.

south of a70r3 is north of a71r3.

southwest of a72r1 is northeast of a73r1.

north of a74r0 is south of a75r0.

north of a76r0 is south of a77r0.

northwest of a78r2 is southeast of a79r2.

northeast of a80r2 is southwest of a81r2.

north of a26r0 is south of a82r0.

south of a20r0 is north of a83r0.

northwest of a16r4 is southeast of a63r4.

northwest of a18r2 is southeast of a65r2.

southeast of a29r4 is northwest of a67r4.

northeast of a29r4 is southwest of a69r4.

east of a11r3 is west of a71r3.

northeast of a9r1 is southwest of a73r1.

southwest of a5r3 is northeast of a8r3.

southwest of a14r1 is northeast of a43r1.

southwest of a13r3 is northeast of a27r3.

southwest of a23r0 is northeast of a75r0.

south of a24r0 is north of a77r0.

southeast of a28r2 is northwest of a79r2.

southwest of a18r2 is northeast of a81r2.

southwest of a16r4 is northeast of a59r4.

northeast of a28r2 is southwest of a61r2.

northwest of a14r1 is southeast of a57r1.

northwest of a15r3 is southeast of a31r3.

northeast of a6r3 is southwest of a33r3.

north of a5r3 is south of a35r3.

north of a3r3 is south of a37r3.

north of a12r3 is south of a39r3.

north of a13r3 is south of a41r3.

southeast of a6r3 is northwest of a45r3.

southwest of a0r3 is northeast of a47r3.

southeast of a2r3 is northwest of a49r3.

southeast of a12r3 is northwest of a27r3.

southwest of a15r3 is northeast of a51r3.

northwest of a0r3 is southeast of a53r3.

northeast of a2r3 is southwest of a55r3.

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
the player is in a30r3.

the orientation of the player is 6.
