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
the X of the a2r0 is "547448.0".
the Y of the a2r0 is "1085714.0".

[create a3r0]
a3r0 is a area. "An area (3) in r0".
the printed name of the a3r0 is "Room 0".
Understand "Area 3 in Room 0" as a3r0.
the parent of the a3r0 is r0.
the X of the a3r0 is "547479.0876068377".
the Y of the a3r0 is "1085688.166239316".

[create a4r0]
a4r0 is a area. "An area (4) in r0".
the printed name of the a4r0 is "Room 0".
Understand "Area 4 in Room 0" as a4r0.
the parent of the a4r0 is r0.
the X of the a4r0 is "547741.0".
the Y of the a4r0 is "1085704.0".

[create a5r0]
a5r0 is a area. "An area (5) in r0".
the printed name of the a5r0 is "Room 0".
Understand "Area 5 in Room 0" as a5r0.
the parent of the a5r0 is r0.
the X of the a5r0 is "547773.8264604809".
the Y of the a5r0 is "1085705.7673539517".

[create a8r0]
a8r0 is a area. "An area (8) in r0".
the printed name of the a8r0 is "Room 0".
Understand "Area 8 in Room 0" as a8r0.
the parent of the a8r0 is r0.
the X of the a8r0 is "548208.0".
the Y of the a8r0 is "1085704.0".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "548175.0059829059".
the Y of the a9r0 is "1085705.7505982905".

[create a16r0]
a16r0 is a area. "An area (16) in r0".
the printed name of the a16r0 is "Room 0".
Understand "Area 16 in Room 0" as a16r0.
the parent of the a16r0 is r0.
the X of the a16r0 is "547984.0".
the Y of the a16r0 is "1085492.0".

[create a17r0]
a17r0 is a area. "An area (17) in r0".
the printed name of the a17r0 is "Room 0".
Understand "Area 17 in Room 0" as a17r0.
the parent of the a17r0 is r0.
the X of the a17r0 is "547982.4644194755".
the Y of the a17r0 is "1085516.2322097376".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "547439.0".
the Y of the a22r0 is "1085624.0".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "547479.7622699386".
the Y of the a23r0 is "1085642.6487730062".

[create a51r0]
a51r0 is a area. "An area (51) in r0".
the printed name of the a51r0 is "Room 0".
Understand "Area 51 in Room 0" as a51r0.
the parent of the a51r0 is r0.
the X of the a51r0 is "549317.8043478262".
the Y of the a51r0 is "1085575.0".

[create a52r0]
a52r0 is a area. "An area (52) in r0".
the printed name of the a52r0 is "Room 0".
Understand "Area 52 in Room 0" as a52r0.
the parent of the a52r0 is r0.
the X of the a52r0 is "549281.1750569604".
the Y of the a52r0 is "1085583.5637036283".

[create a55r0]
a55r0 is a area. "An area (55) in r0".
the printed name of the a55r0 is "Room 0".
Understand "Area 55 in Room 0" as a55r0.
the parent of the a55r0 is r0.
the X of the a55r0 is "548149.6485044326".
the Y of the a55r0 is "1085534.144790269".

[create a56r0]
a56r0 is a area. "An area (56) in r0".
the printed name of the a56r0 is "Room 0".
Understand "Area 56 in Room 0" as a56r0.
the parent of the a56r0 is r0.
the X of the a56r0 is "548150.1139460726".
the Y of the a56r0 is "1085539.5596406544".

[create a57r0]
a57r0 is a area. "An area (57) in r0".
the printed name of the a57r0 is "Room 0".
Understand "Area 57 in Room 0" as a57r0.
the parent of the a57r0 is r0.
the X of the a57r0 is "548158.9253140043".
the Y of the a57r0 is "1085547.3560096084".

[create a60r0]
a60r0 is a area. "An area (60) in r0".
the printed name of the a60r0 is "Room 0".
Understand "Area 60 in Room 0" as a60r0.
the parent of the a60r0 is r0.
the X of the a60r0 is "547495.5405048297".
the Y of the a60r0 is "1085660.8544286496".

[create a71r0]
a71r0 is a area. "An area (71) in r0".
the printed name of the a71r0 is "Room 0".
Understand "Area 71 in Room 0" as a71r0.
the parent of the a71r0 is r0.
the X of the a71r0 is "547783.4944995979".
the Y of the a71r0 is "1085611.0205706106".

[create a75r0]
a75r0 is a area. "An area (75) in r0".
the printed name of the a75r0 is "Room 0".
Understand "Area 75 in Room 0" as a75r0.
the parent of the a75r0 is r0.
the X of the a75r0 is "547835.7205422446".
the Y of the a75r0 is "1085576.2225953648".

[create a10r1]
a10r1 is a area. "An area (10) in r1".
the printed name of the a10r1 is "Room 1".
Understand "Area 10 in Room 1" as a10r1.
the parent of the a10r1 is r1.
the X of the a10r1 is "549834.0".
the Y of the a10r1 is "1084227.0".

[create a11r1]
a11r1 is a area. "An area (11) in r1".
the printed name of the a11r1 is "Room 1".
Understand "Area 11 in Room 1" as a11r1.
the parent of the a11r1 is r1.
the X of the a11r1 is "549745.3368675597".
the Y of the a11r1 is "1084262.2857142857".

[create a26r1]
a26r1 is a area. "An area (26) in r1".
the printed name of the a26r1 is "Room 1".
Understand "Area 26 in Room 1" as a26r1.
the parent of the a26r1 is r1.
the X of the a26r1 is "547720.0".
the Y of the a26r1 is "1083591.0".

[create a27r1]
a27r1 is a area. "An area (27) in r1".
the printed name of the a27r1 is "Room 1".
Understand "Area 27 in Room 1" as a27r1.
the parent of the a27r1 is r1.
the X of the a27r1 is "547751.2137161084".
the Y of the a27r1 is "1083616.6339712918".

[create a30r1]
a30r1 is a area. "An area (30) in r1".
the printed name of the a30r1 is "Room 1".
Understand "Area 30 in Room 1" as a30r1.
the parent of the a30r1 is r1.
the X of the a30r1 is "547711.0".
the Y of the a30r1 is "1083680.0".

[create a31r1]
a31r1 is a area. "An area (31) in r1".
the printed name of the a31r1 is "Room 1".
Understand "Area 31 in Room 1" as a31r1.
the parent of the a31r1 is r1.
the X of the a31r1 is "547742.0059757694".
the Y of the a31r1 is "1083660.657907662".

[create a36r1]
a36r1 is a area. "An area (36) in r1".
the printed name of the a36r1 is "Room 1".
Understand "Area 36 in Room 1" as a36r1.
the parent of the a36r1 is r1.
the X of the a36r1 is "549226.0".
the Y of the a36r1 is "1083600.0".

[create a37r1]
a37r1 is a area. "An area (37) in r1".
the printed name of the a37r1 is "Room 1".
Understand "Area 37 in Room 1" as a37r1.
the parent of the a37r1 is r1.
the X of the a37r1 is "549189.875".
the Y of the a37r1 is "1083685.7209595959".

[create a61r1]
a61r1 is a area. "An area (61) in r1".
the printed name of the a61r1 is "Room 1".
Understand "Area 61 in Room 1" as a61r1.
the parent of the a61r1 is r1.
the X of the a61r1 is "547765.389199255".
the Y of the a61r1 is "1083640.2597765364".

[create a64r1]
a64r1 is a area. "An area (64) in r1".
the printed name of the a64r1 is "Room 1".
Understand "Area 64 in Room 1" as a64r1.
the parent of the a64r1 is r1.
the X of the a64r1 is "549168.4184328842".
the Y of the a64r1 is "1083736.1969348686".

[create a67r1]
a67r1 is a area. "An area (67) in r1".
the printed name of the a67r1 is "Room 1".
Understand "Area 67 in Room 1" as a67r1.
the parent of the a67r1 is r1.
the X of the a67r1 is "549649.0324175665".
the Y of the a67r1 is "1084302.013516861".

[create a80r1]
a80r1 is a area. "An area (80) in r1".
the printed name of the a80r1 is "Room 1".
Understand "Area 80 in Room 1" as a80r1.
the parent of the a80r1 is r1.
the X of the a80r1 is "548149.1865194305".
the Y of the a80r1 is "1083770.3208574704".

[create a46r2]
a46r2 is a area. "An area (46) in r2".
the printed name of the a46r2 is "Room 2".
Understand "Area 46 in Room 2" as a46r2.
the parent of the a46r2 is r2.
the X of the a46r2 is "549511.0".
the Y of the a46r2 is "1085475.677419355".

[create a47r2]
a47r2 is a area. "An area (47) in r2".
the printed name of the a47r2 is "Room 2".
Understand "Area 47 in Room 2" as a47r2.
the parent of the a47r2 is r2.
the X of the a47r2 is "549533.3295318402".
the Y of the a47r2 is "1085439.0241936338".

[create a12r3]
a12r3 is a area. "An area (12) in r3".
the printed name of the a12r3 is "Room 3".
Understand "Area 12 in Room 3" as a12r3.
the parent of the a12r3 is r3.
the X of the a12r3 is "546657.0063081433".
the Y of the a12r3 is "1084562.6664967504".

[create a13r3]
a13r3 is a area. "An area (13) in r3".
the printed name of the a13r3 is "Room 3".
Understand "Area 13 in Room 3" as a13r3.
the parent of the a13r3 is r3.
the X of the a13r3 is "546670.3371097946".
the Y of the a13r3 is "1084607.1437413543".

[create a28r3]
a28r3 is a area. "An area (28) in r3".
the printed name of the a28r3 is "Room 3".
Understand "Area 28 in Room 3" as a28r3.
the parent of the a28r3 is r3.
the X of the a28r3 is "545089.0".
the Y of the a28r3 is "1083680.0".

[create a29r3]
a29r3 is a area. "An area (29) in r3".
the printed name of the a29r3 is "Room 3".
Understand "Area 29 in Room 3" as a29r3.
the parent of the a29r3 is r3.
the X of the a29r3 is "545120.0059757696".
the Y of the a29r3 is "1083660.657907662".

[create a32r3]
a32r3 is a area. "An area (32) in r3".
the printed name of the a32r3 is "Room 3".
Understand "Area 32 in Room 3" as a32r3.
the parent of the a32r3 is r3.
the X of the a32r3 is "546385.0".
the Y of the a32r3 is "1083600.0".

[create a33r3]
a33r3 is a area. "An area (33) in r3".
the printed name of the a33r3 is "Room 3".
Understand "Area 33 in Room 3" as a33r3.
the parent of the a33r3 is r3.
the X of the a33r3 is "546349.0".
the Y of the a33r3 is "1083694.8699839488".

[create a34r3]
a34r3 is a area. "An area (34) in r3".
the printed name of the a34r3 is "Room 3".
Understand "Area 34 in Room 3" as a34r3.
the parent of the a34r3 is r3.
the X of the a34r3 is "547297.0".
the Y of the a34r3 is "1083600.0".

[create a35r3]
a35r3 is a area. "An area (35) in r3".
the printed name of the a35r3 is "Room 3".
Understand "Area 35 in Room 3" as a35r3.
the parent of the a35r3 is r3.
the X of the a35r3 is "547254.4517163367".
the Y of the a35r3 is "1083649.4252993807".

[create a38r3]
a38r3 is a area. "An area (38) in r3".
the printed name of the a38r3 is "Room 3".
Understand "Area 38 in Room 3" as a38r3.
the parent of the a38r3 is r3.
the X of the a38r3 is "546773.0".
the Y of the a38r3 is "1083956.0".

[create a39r3]
a39r3 is a area. "An area (39) in r3".
the printed name of the a39r3 is "Room 3".
Understand "Area 39 in Room 3" as a39r3.
the parent of the a39r3 is r3.
the X of the a39r3 is "546691.2528429537".
the Y of the a39r3 is "1083989.4611872144".

[create a40r3]
a40r3 is a area. "An area (40) in r3".
the printed name of the a40r3 is "Room 3".
Understand "Area 40 in Room 3" as a40r3.
the parent of the a40r3 is r3.
the X of the a40r3 is "546773.0".
the Y of the a40r3 is "1084223.689497717".

[create a41r3]
a41r3 is a area. "An area (41) in r3".
the printed name of the a41r3 is "Room 3".
Understand "Area 41 in Room 3" as a41r3.
the parent of the a41r3 is r3.
the X of the a41r3 is "546793.407255664".
the Y of the a41r3 is "1084190.2283105021".

[create a42r3]
a42r3 is a area. "An area (42) in r3".
the printed name of the a42r3 is "Room 3".
Understand "Area 42 in Room 3" as a42r3.
the parent of the a42r3 is r3.
the X of the a42r3 is "545098.0".
the Y of the a42r3 is "1083591.0".

[create a43r3]
a43r3 is a area. "An area (43) in r3".
the printed name of the a43r3 is "Room 3".
Understand "Area 43 in Room 3" as a43r3.
the parent of the a43r3 is r3.
the X of the a43r3 is "545129.2137161086".
the Y of the a43r3 is "1083616.6339712918".

[create a44r3]
a44r3 is a area. "An area (44) in r3".
the printed name of the a44r3 is "Room 3".
Understand "Area 44 in Room 3" as a44r3.
the parent of the a44r3 is r3.
the X of the a44r3 is "546996.0".
the Y of the a44r3 is "1083572.0".

[create a45r3]
a45r3 is a area. "An area (45) in r3".
the printed name of the a45r3 is "Room 3".
Understand "Area 45 in Room 3" as a45r3.
the parent of the a45r3 is r3.
the X of the a45r3 is "547029.1336316653".
the Y of the a45r3 is "1083623.7822095975".

[create a53r3]
a53r3 is a area. "An area (53) in r3".
the printed name of the a53r3 is "Room 3".
Understand "Area 53 in Room 3" as a53r3.
the parent of the a53r3 is r3.
the X of the a53r3 is "546772.9227278628".
the Y of the a53r3 is "1084642.2472575372".

[create a62r3]
a62r3 is a area. "An area (62) in r3".
the printed name of the a62r3 is "Room 3".
Understand "Area 62 in Room 3" as a62r3.
the parent of the a62r3 is r3.
the X of the a62r3 is "546345.5625713556".
the Y of the a62r3 is "1083703.4828669569".

[create a63r3]
a63r3 is a area. "An area (63) in r3".
the printed name of the a63r3 is "Room 3".
Understand "Area 63 in Room 3" as a63r3.
the parent of the a63r3 is r3.
the X of the a63r3 is "547190.9629832632".
the Y of the a63r3 is "1083728.2779362001".

[create a65r3]
a65r3 is a area. "An area (65) in r3".
the printed name of the a65r3 is "Room 3".
Understand "Area 65 in Room 3" as a65r3.
the parent of the a65r3 is r3.
the X of the a65r3 is "546626.0654209425".
the Y of the a65r3 is "1084020.1392351503".

[create a66r3]
a66r3 is a area. "An area (66) in r3".
the printed name of the a66r3 is "Room 3".
Understand "Area 66 in Room 3" as a66r3.
the parent of the a66r3 is r3.
the X of the a66r3 is "546914.1932909233".
the Y of the a66r3 is "1084231.5597819425".

[create a68r3]
a68r3 is a area. "An area (68) in r3".
the printed name of the a68r3 is "Room 3".
Understand "Area 68 in Room 3" as a68r3.
the parent of the a68r3 is r3.
the X of the a68r3 is "546664.6436919513".
the Y of the a68r3 is "1084375.9456187894".

[create a69r3]
a69r3 is a area. "An area (69) in r3".
the printed name of the a69r3 is "Room 3".
Understand "Area 69 in Room 3" as a69r3.
the parent of the a69r3 is r3.
the X of the a69r3 is "546790.3084255729".
the Y of the a69r3 is "1084418.9465198256".

[create a70r3]
a70r3 is a area. "An area (70) in r3".
the printed name of the a70r3 is "Room 3".
Understand "Area 70 in Room 3" as a70r3.
the parent of the a70r3 is r3.
the X of the a70r3 is "545143.389199255".
the Y of the a70r3 is "1083640.2597765364".

[create a79r3]
a79r3 is a area. "An area (79) in r3".
the printed name of the a79r3 is "Room 3".
Understand "Area 79 in Room 3" as a79r3.
the parent of the a79r3 is r3.
the X of the a79r3 is "547100.1286841574".
the Y of the a79r3 is "1083733.4637773987".

[create a81r3]
a81r3 is a area. "An area (81) in r3".
the printed name of the a81r3 is "Room 3".
Understand "Area 81 in Room 3" as a81r3.
the parent of the a81r3 is r3.
the X of the a81r3 is "545527.6402035004".
the Y of the a81r3 is "1083770.6825207148".

[create a0r4]
a0r4 is a area. "An area (0) in r4".
the printed name of the a0r4 is "Room 4".
Understand "Area 0 in Room 4" as a0r4.
the parent of the a0r4 is r4.
the X of the a0r4 is "546901.0".
the Y of the a0r4 is "1085714.0".

[create a1r4]
a1r4 is a area. "An area (1) in r4".
the printed name of the a1r4 is "Room 4".
Understand "Area 1 in Room 4" as a1r4.
the parent of the a1r4 is r4.
the X of the a1r4 is "546870.0846219934".
the Y of the a1r4 is "1085688.1834621995".

[create a6r4]
a6r4 is a area. "An area (6) in r4".
the printed name of the a6r4 is "Room 4".
Understand "Area 6 in Room 4" as a6r4.
the parent of the a6r4 is r4.
the X of the a6r4 is "546086.0".
the Y of the a6r4 is "1085704.0".

[create a7r4]
a7r4 is a area. "An area (7) in r4".
the printed name of the a7r4 is "Room 4".
Understand "Area 7 in Room 4" as a7r4.
the parent of the a7r4 is r4.
the X of the a7r4 is "546118.9940170939".
the Y of the a7r4 is "1085705.7505982905".

[create a18r4]
a18r4 is a area. "An area (18) in r4".
the printed name of the a18r4 is "Room 4".
Understand "Area 18 in Room 4" as a18r4.
the parent of the a18r4 is r4.
the X of the a18r4 is "545111.0".
the Y of the a18r4 is "1085624.0".

[create a19r4]
a19r4 is a area. "An area (19) in r4".
the printed name of the a19r4 is "Room 4".
Understand "Area 19 in Room 4" as a19r4.
the parent of the a19r4 is r4.
the X of the a19r4 is "545153.06".
the Y of the a19r4 is "1085642.5189999999".

[create a20r4]
a20r4 is a area. "An area (20) in r4".
the printed name of the a20r4 is "Room 4".
Understand "Area 20 in Room 4" as a20r4.
the parent of the a20r4 is r4.
the X of the a20r4 is "546910.0".
the Y of the a20r4 is "1085624.0".

[create a21r4]
a21r4 is a area. "An area (21) in r4".
the printed name of the a21r4 is "Room 4".
Understand "Area 21 in Room 4" as a21r4.
the parent of the a21r4 is r4.
the X of the a21r4 is "546869.5982564336".
the Y of the a21r4 is "1085642.6848256432".

[create a24r4]
a24r4 is a area. "An area (24) in r4".
the printed name of the a24r4 is "Room 4".
Understand "Area 24 in Room 4" as a24r4.
the parent of the a24r4 is r4.
the X of the a24r4 is "545120.0".
the Y of the a24r4 is "1085714.0".

[create a25r4]
a25r4 is a area. "An area (25) in r4".
the printed name of the a25r4 is "Room 4".
Understand "Area 25 in Room 4" as a25r4.
the parent of the a25r4 is r4.
the X of the a25r4 is "545150.9153780071".
the Y of the a25r4 is "1085688.1834621993".

[create a58r4]
a58r4 is a area. "An area (58) in r4".
the printed name of the a58r4 is "Room 4".
Understand "Area 58 in Room 4" as a58r4.
the parent of the a58r4 is r4.
the X of the a58r4 is "545168.092671901".
the Y of the a58r4 is "1085659.8065726862".

[create a59r4]
a59r4 is a area. "An area (59) in r4".
the printed name of the a59r4 is "Room 4".
Understand "Area 59 in Room 4" as a59r4.
the parent of the a59r4 is r4.
the X of the a59r4 is "546853.6927655564".
the Y of the a59r4 is "1085661.104115366".

[create a72r4]
a72r4 is a area. "An area (72) in r4".
the printed name of the a72r4 is "Room 4".
Understand "Area 72 in Room 4" as a72r4.
the parent of the a72r4 is r4.
the X of the a72r4 is "546125.2410188394".
the Y of the a72r4 is "1085644.2176310967".

[create a73r4]
a73r4 is a area. "An area (73) in r4".
the printed name of the a73r4 is "Room 4".
Understand "Area 73 in Room 4" as a73r4.
the parent of the a73r4 is r4.
the X of the a73r4 is "545526.3445626479".
the Y of the a73r4 is "1085534.3503250591".

[create a74r4]
a74r4 is a area. "An area (74) in r4".
the printed name of the a74r4 is "Room 4".
Understand "Area 74 in Room 4" as a74r4.
the parent of the a74r4 is r4.
the X of the a74r4 is "546484.727374957".
the Y of the a74r4 is "1085547.9140034495".

[create a76r4]
a76r4 is a area. "An area (76) in r4".
the printed name of the a76r4 is "Room 4".
Understand "Area 76 in Room 4" as a76r4.
the parent of the a76r4 is r4.
the X of the a76r4 is "545527.3784065933".
the Y of the a76r4 is "1084679.5".

[create a77r4]
a77r4 is a area. "An area (77) in r4".
the printed name of the a77r4 is "Room 4".
Understand "Area 77 in Room 4" as a77r4.
the parent of the a77r4 is r4.
the X of the a77r4 is "546224.6801725186".
the Y of the a77r4 is "1084727.2069072586".

[create a78r4]
a78r4 is a area. "An area (78) in r4".
the printed name of the a78r4 is "Room 4".
Understand "Area 78 in Room 4" as a78r4.
the parent of the a78r4 is r4.
the X of the a78r4 is "546549.695783145".
the Y of the a78r4 is "1085120.9554716148".

[create a14r5]
a14r5 is a area. "An area (14) in r5".
the printed name of the a14r5 is "Room 5".
Understand "Area 14 in Room 5" as a14r5.
the parent of the a14r5 is r5.
the X of the a14r5 is "549511.0".
the Y of the a14r5 is "1085575.0".

[create a15r5]
a15r5 is a area. "An area (15) in r5".
the printed name of the a15r5 is "Room 5".
Understand "Area 15 in Room 5" as a15r5.
the parent of the a15r5 is r5.
the X of the a15r5 is "549478.8007246377".
the Y of the a15r5 is "1085550.1693548388".

[create a46r5]
a46r5 is a area. "An area (46) in r5".
the printed name of the a46r5 is "Room 5".
Understand "Area 46 in Room 5" as a46r5.
the parent of the a46r5 is r5.
the X of the a46r5 is "549511.0".
the Y of the a46r5 is "1085475.677419355".

[create a48r5]
a48r5 is a area. "An area (48) in r5".
the printed name of the a48r5 is "Room 5".
Understand "Area 48 in Room 5" as a48r5.
the parent of the a48r5 is r5.
the X of the a48r5 is "549495.7152467689".
the Y of the a48r5 is "1085500.7668620849".

[create a49r5]
a49r5 is a area. "An area (49) in r5".
the printed name of the a49r5 is "Room 5".
Understand "Area 49 in Room 5" as a49r5.
the parent of the a49r5 is r5.
the X of the a49r5 is "549336.0".
the Y of the a49r5 is "1085566.0".

[create a50r5]
a50r5 is a area. "An area (50) in r5".
the printed name of the a50r5 is "Room 5".
Understand "Area 50 in Room 5" as a50r5.
the parent of the a50r5 is r5.
the X of the a50r5 is "549414.4021739131".
the Y of the a50r5 is "1085563.552540156".

[create a51r5]
a51r5 is a area. "An area (51) in r5".
the printed name of the a51r5 is "Room 5".
Understand "Area 51 in Room 5" as a51r5.
the parent of the a51r5 is r5.
the X of the a51r5 is "549317.8043478262".
the Y of the a51r5 is "1085575.0".

[create a54r5]
a54r5 is a area. "An area (54) in r5".
the printed name of the a54r5 is "Room 5".
Understand "Area 54 in Room 5" as a54r5.
the parent of the a54r5 is r5.
the X of the a54r5 is "549472.8286071355".
the Y of the a54r5 is "1085542.424978698".

[create d0]
d0 is a door. "A door between a75r0 and a63r3".
d0 is south of a75r0 and north of a63r3.

[create d2]
d2 is a door. "A door between a55r0 and a80r1".
d2 is south of a55r0 and north of a80r1.

[create d4]
d4 is a door. "A door between a53r3 and a78r4".
d4 is northwest of a53r3 and southeast of a78r4.

[create d5]
d5 is a door. "A door between a81r3 and a76r4".
d5 is north of a81r3 and south of a76r4.

[create d6]
d6 is a door. "A door between a67r1 and a47r2".
d6 is north of a67r1 and south of a47r2.

[create d8]
d8 is a door. "A door between a68r3 and a77r4".
d8 is northwest of a68r3 and southeast of a77r4.

southwest of a0r4 is northeast of a1r4.

southeast of a2r0 is northwest of a3r0.

east of a4r0 is west of a5r0.

east of a6r4 is west of a7r4.

west of a8r0 is east of a9r0.

west of a10r1 is east of a11r1.

north of a12r3 is south of a13r3.

southwest of a14r5 is northeast of a15r5.

north of a16r0 is south of a17r0.

northeast of a18r4 is southwest of a19r4.

northwest of a20r4 is southeast of a21r4.

northeast of a22r0 is southwest of a23r0.

southeast of a24r4 is northwest of a25r4.

northeast of a26r1 is southwest of a27r1.

southeast of a28r3 is northwest of a29r3.

southeast of a30r1 is northwest of a31r1.

north of a32r3 is south of a33r3.

northwest of a34r3 is southeast of a35r3.

northwest of a36r1 is southeast of a37r1.

west of a38r3 is east of a39r3.

southeast of a40r3 is northwest of a41r3.

northeast of a42r3 is southwest of a43r3.

northeast of a44r3 is southwest of a45r3.

southeast of a46r2 is northwest of a47r2.

east of a46r2 is west of a46r2.

east of a49r5 is west of a50r5.

west of a51r0 is east of a52r0.

east of a13r3 is west of a53r3.

southwest of a15r5 is northeast of a54r5.

east of a17r0 is west of a55r0.

north of a55r0 is south of a56r0.

northeast of a56r0 is southwest of a57r0.

northeast of a19r4 is southwest of a58r4.

northwest of a21r4 is southeast of a59r4.

northeast of a23r0 is southwest of a60r0.

southeast of a25r4 is northwest of a58r4.

southeast of a31r1 is northwest of a61r1.

north of a33r3 is south of a62r3.

northwest of a35r3 is southeast of a63r3.

northwest of a37r1 is southeast of a64r1.

northwest of a39r3 is southeast of a65r3.

east of a41r3 is west of a66r3.

west of a11r1 is east of a67r1.

east of a68r3 is west of a69r3.

northeast of a43r3 is southwest of a70r3.

northeast of a27r1 is southwest of a61r1.

southeast of a29r3 is northwest of a70r3.

southwest of a1r4 is northeast of a59r4.

southeast of a3r0 is northwest of a60r0.

south of a5r0 is north of a71r0.

south of a7r4 is north of a72r4.

south of a9r0 is north of a57r0.

east of a58r4 is west of a73r4.

west of a72r4 is east of a73r4.

east of a60r0 is west of a71r0.

east of a72r4 is west of a74r4.

west of a59r4 is east of a74r4.

west of a56r0 is east of a75r0.

southeast of a71r0 is northwest of a75r0.

east of a76r4 is west of a77r4.

northeast of a77r4 is southwest of a78r4.

south of a73r4 is north of a76r4.

south of a74r4 is north of a78r4.

northwest of a48r5 is southeast of a54r5.

east of a50r5 is west of a54r5.

west of a52r0 is east of a57r0.

south of a66r3 is north of a79r3.

north of a65r3 is south of a68r3.

northwest of a66r3 is southeast of a69r3.

south of a53r3 is north of a69r3.

east of a61r1 is west of a80r1.

west of a62r3 is east of a81r3.

west of a64r1 is east of a80r1.

northeast of a62r3 is southwest of a65r3.

west of a63r3 is east of a79r3.

northeast of a45r3 is southwest of a79r3.

northeast of a64r1 is southwest of a67r1.

east of a70r3 is west of a81r3.

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
the player is in a76r4.

the orientation of the player is 6.
