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

[create a20r0]
a20r0 is a area. "An area (20) in r0".
the printed name of the a20r0 is "Room 0".
Understand "Area 20 in Room 0" as a20r0.
the parent of the a20r0 is r0.
the X of the a20r0 is "548935.7496886398".
the Y of the a20r0 is "1752562.3173736057".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "548921.9861188938".
the Y of the a22r0 is "1752564.0131715976".

[create a31r0]
a31r0 is a area. "An area (31) in r0".
the printed name of the a31r0 is "Room 0".
Understand "Area 31 in Room 0" as a31r0.
the parent of the a31r0 is r0.
the X of the a31r0 is "549029.2968536066".
the Y of the a31r0 is "1752451.4209743899".

[create a41r0]
a41r0 is a area. "An area (41) in r0".
the printed name of the a41r0 is "Room 0".
Understand "Area 41 in Room 0" as a41r0.
the parent of the a41r0 is r0.
the X of the a41r0 is "548818.8224123432".
the Y of the a41r0 is "1752420.092132829".

[create a115r0]
a115r0 is a area. "An area (115) in r0".
the printed name of the a115r0 is "Room 0".
Understand "Area 115 in Room 0" as a115r0.
the parent of the a115r0 is r0.
the X of the a115r0 is "548788.0".
the Y of the a115r0 is "1752372.0".

[create a117r0]
a117r0 is a area. "An area (117) in r0".
the printed name of the a117r0 is "Room 0".
Understand "Area 117 in Room 0" as a117r0.
the parent of the a117r0 is r0.
the X of the a117r0 is "549070.0".
the Y of the a117r0 is "1752400.0".

[create a15r1]
a15r1 is a area. "An area (15) in r1".
the printed name of the a15r1 is "Room 1".
Understand "Area 15 in Room 1" as a15r1.
the parent of the a15r1 is r1.
the X of the a15r1 is "549509.3333333328".
the Y of the a15r1 is "1754460.9694444444".

[create a16r1]
a16r1 is a area. "An area (16) in r1".
the printed name of the a16r1 is "Room 1".
Understand "Area 16 in Room 1" as a16r1.
the parent of the a16r1 is r1.
the X of the a16r1 is "549576.5167685037".
the Y of the a16r1 is "1754373.513395759".

[create a21r1]
a21r1 is a area. "An area (21) in r1".
the printed name of the a21r1 is "Room 1".
Understand "Area 21 in Room 1" as a21r1.
the parent of the a21r1 is r1.
the X of the a21r1 is "549581.8708448358".
the Y of the a21r1 is "1754205.3162913683".

[create a69r1]
a69r1 is a area. "An area (69) in r1".
the printed name of the a69r1 is "Room 1".
Understand "Area 69 in Room 1" as a69r1.
the parent of the a69r1 is r1.
the X of the a69r1 is "549676.0".
the Y of the a69r1 is "1754504.0".

[create a70r1]
a70r1 is a area. "An area (70) in r1".
the printed name of the a70r1 is "Room 1".
Understand "Area 70 in Room 1" as a70r1.
the parent of the a70r1 is r1.
the X of the a70r1 is "549642.6666666669".
the Y of the a70r1 is "1754453.5364495008".

[create a119r1]
a119r1 is a area. "An area (119) in r1".
the printed name of the a119r1 is "Room 1".
Understand "Area 119 in Room 1" as a119r1.
the parent of the a119r1 is r1.
the X of the a119r1 is "549476.0".
the Y of the a119r1 is "1754504.0".

[create a23r2]
a23r2 is a area. "An area (23) in r2".
the printed name of the a23r2 is "Room 2".
Understand "Area 23 in Room 2" as a23r2.
the parent of the a23r2 is r2.
the X of the a23r2 is "550176.2390740267".
the Y of the a23r2 is "1752562.4312827692".

[create a32r2]
a32r2 is a area. "An area (32) in r2".
the printed name of the a32r2 is "Room 2".
Understand "Area 32 in Room 2" as a32r2.
the parent of the a32r2 is r2.
the X of the a32r2 is "550198.289651646".
the Y of the a32r2 is "1752560.8833195833".

[create a33r2]
a33r2 is a area. "An area (33) in r2".
the printed name of the a33r2 is "Room 2".
Understand "Area 33 in Room 2" as a33r2.
the parent of the a33r2 is r2.
the X of the a33r2 is "550292.4539705467".
the Y of the a33r2 is "1752448.7643334023".

[create a42r2]
a42r2 is a area. "An area (42) in r2".
the printed name of the a42r2 is "Room 2".
Understand "Area 42 in Room 2" as a42r2.
the parent of the a42r2 is r2.
the X of the a42r2 is "550080.7043713275".
the Y of the a42r2 is "1752422.7067112252".

[create a116r2]
a116r2 is a area. "An area (116) in r2".
the printed name of the a116r2 is "Room 2".
Understand "Area 116 in Room 2" as a116r2.
the parent of the a116r2 is r2.
the X of the a116r2 is "550050.0".
the Y of the a116r2 is "1752372.0".

[create a118r2]
a118r2 is a area. "An area (118) in r2".
the printed name of the a118r2 is "Room 2".
Understand "Area 118 in Room 2" as a118r2.
the parent of the a118r2 is r2.
the X of the a118r2 is "550333.0".
the Y of the a118r2 is "1752400.0".

[create a6r3]
a6r3 is a area. "An area (6) in r3".
the printed name of the a6r3 is "Room 3".
Understand "Area 6 in Room 3" as a6r3.
the parent of the a6r3 is r3.
the X of the a6r3 is "550708.9486065779".
the Y of the a6r3 is "1754378.1064398696".

[create a7r3]
a7r3 is a area. "An area (7) in r3".
the printed name of the a7r3 is "Room 3".
Understand "Area 7 in Room 3" as a7r3.
the parent of the a7r3 is r3.
the X of the a7r3 is "551038.5509290382".
the Y of the a7r3 is "1754461.0062088708".

[create a19r3]
a19r3 is a area. "An area (19) in r3".
the printed name of the a19r3 is "Room 3".
Understand "Area 19 in Room 3" as a19r3.
the parent of the a19r3 is r3.
the X of the a19r3 is "550374.4921500897".
the Y of the a19r3 is "1754460.7986347189".

[create a45r3]
a45r3 is a area. "An area (45) in r3".
the printed name of the a45r3 is "Room 3".
Understand "Area 45 in Room 3" as a45r3.
the parent of the a45r3 is r3.
the X of the a45r3 is "550327.0".
the Y of the a45r3 is "1754514.0".

[create a46r3]
a46r3 is a area. "An area (46) in r3".
the printed name of the a46r3 is "Room 3".
Understand "Area 46 in Room 3" as a46r3.
the parent of the a46r3 is r3.
the X of the a46r3 is "550357.9153780069".
the Y of the a46r3 is "1754488.1834621993".

[create a53r3]
a53r3 is a area. "An area (53) in r3".
the printed name of the a53r3 is "Room 3".
Understand "Area 53 in Room 3" as a53r3.
the parent of the a53r3 is r3.
the X of the a53r3 is "551086.0".
the Y of the a53r3 is "1754514.0".

[create a54r3]
a54r3 is a area. "An area (54) in r3".
the printed name of the a54r3 is "Room 3".
Understand "Area 54 in Room 3" as a54r3.
the parent of the a54r3 is r3.
the X of the a54r3 is "551054.9123931623".
the Y of the a54r3 is "1754488.1662393163".

[create a85r3]
a85r3 is a area. "An area (85) in r3".
the printed name of the a85r3 is "Room 3".
Understand "Area 85 in Room 3" as a85r3.
the parent of the a85r3 is r3.
the X of the a85r3 is "550318.0".
the Y of the a85r3 is "1754424.0".

[create a86r3]
a86r3 is a area. "An area (86) in r3".
the printed name of the a86r3 is "Room 3".
Understand "Area 86 in Room 3" as a86r3.
the parent of the a86r3 is r3.
the X of the a86r3 is "550358.7622699386".
the Y of the a86r3 is "1754442.648773006".

[create a87r3]
a87r3 is a area. "An area (87) in r3".
the printed name of the a87r3 is "Room 3".
Understand "Area 87 in Room 3" as a87r3.
the parent of the a87r3 is r3.
the X of the a87r3 is "551095.0".
the Y of the a87r3 is "1754424.0".

[create a88r3]
a88r3 is a area. "An area (88) in r3".
the printed name of the a88r3 is "Room 3".
Understand "Area 88 in Room 3" as a88r3.
the parent of the a88r3 is r3.
the X of the a88r3 is "551054.662577747".
the Y of the a88r3 is "1754442.6912577746".

[create a0r4]
a0r4 is a area. "An area (0) in r4".
the printed name of the a0r4 is "Room 4".
Understand "Area 0 in Room 4" as a0r4.
the parent of the a0r4 is r4.
the X of the a0r4 is "546709.4004109498".
the Y of the a0r4 is "1754373.2254542701".

[create a1r4]
a1r4 is a area. "An area (1) in r4".
the printed name of the a1r4 is "Room 4".
Understand "Area 1 in Room 4" as a1r4.
the parent of the a1r4 is r4.
the X of the a1r4 is "547017.944080346".
the Y of the a1r4 is "1754343.585807492".

[create a2r4]
a2r4 is a area. "An area (2) in r4".
the printed name of the a2r4 is "Room 4".
Understand "Area 2 in Room 4" as a2r4.
the parent of the a2r4 is r4.
the X of the a2r4 is "547023.225697948".
the Y of the a2r4 is "1754350.314791456".

[create a3r4]
a3r4 is a area. "An area (3) in r4".
the printed name of the a3r4 is "Room 4".
Understand "Area 3 in Room 4" as a3r4.
the parent of the a3r4 is r4.
the X of the a3r4 is "547399.9343690763".
the Y of the a3r4 is "1754461.642719334".

[create a4r4]
a4r4 is a area. "An area (4) in r4".
the printed name of the a4r4 is "Room 4".
Understand "Area 4 in Room 4" as a4r4.
the parent of the a4r4 is r4.
the X of the a4r4 is "548073.3701298699".
the Y of the a4r4 is "1754442.9458874457".

[create a5r4]
a5r4 is a area. "An area (5) in r4".
the printed name of the a5r4 is "Room 4".
Understand "Area 5 in Room 4" as a5r4.
the parent of the a5r4 is r4.
the X of the a5r4 is "548227.3619459791".
the Y of the a5r4 is "1754365.4949057796".

[create a8r4]
a8r4 is a area. "An area (8) in r4".
the printed name of the a8r4 is "Room 4".
Understand "Area 8 in Room 4" as a8r4.
the parent of the a8r4 is r4.
the X of the a8r4 is "546645.2247084782".
the Y of the a8r4 is "1754424.6782881601".

[create a9r4]
a9r4 is a area. "An area (9) in r4".
the printed name of the a9r4 is "Room 4".
Understand "Area 9 in Room 4" as a9r4.
the parent of the a9r4 is r4.
the X of the a9r4 is "548545.8869813068".
the Y of the a9r4 is "1754415.9284325426".

[create a10r4]
a10r4 is a area. "An area (10) in r4".
the printed name of the a10r4 is "Room 4".
Understand "Area 10 in Room 4" as a10r4.
the parent of the a10r4 is r4.
the X of the a10r4 is "548811.1653756482".
the Y of the a10r4 is "1754461.2658517454".

[create a11r4]
a11r4 is a area. "An area (11) in r4".
the printed name of the a11r4 is "Room 4".
Understand "Area 11 in Room 4" as a11r4.
the parent of the a11r4 is r4.
the X of the a11r4 is "545503.5427972081".
the Y of the a11r4 is "1754338.7985219033".

[create a12r4]
a12r4 is a area. "An area (12) in r4".
the printed name of the a12r4 is "Room 4".
Understand "Area 12 in Room 4" as a12r4.
the parent of the a12r4 is r4.
the X of the a12r4 is "545883.9343690765".
the Y of the a12r4 is "1754461.642719334".

[create a13r4]
a13r4 is a area. "An area (13) in r4".
the printed name of the a13r4 is "Room 4".
Understand "Area 13 in Room 4" as a13r4.
the parent of the a13r4 is r4.
the X of the a13r4 is "545135.4913012546".
the Y of the a13r4 is "1754462.596106584".

[create a14r4]
a14r4 is a area. "An area (14) in r4".
the printed name of the a14r4 is "Room 4".
Understand "Area 14 in Room 4" as a14r4.
the parent of the a14r4 is r4.
the X of the a14r4 is "548432.3787096875".
the Y of the a14r4 is "1754343.728237071".

[create a17r4]
a17r4 is a area. "An area (17) in r4".
the printed name of the a17r4 is "Room 4".
Understand "Area 17 in Room 4" as a17r4.
the parent of the a17r4 is r4.
the X of the a17r4 is "547874.958925213".
the Y of the a17r4 is "1754461.6795222142".

[create a18r4]
a18r4 is a area. "An area (18) in r4".
the printed name of the a18r4 is "Room 4".
Understand "Area 18 in Room 4" as a18r4.
the parent of the a18r4 is r4.
the X of the a18r4 is "546358.958925213".
the Y of the a18r4 is "1754461.6795222145".

[create a24r4]
a24r4 is a area. "An area (24) in r4".
the printed name of the a24r4 is "Room 4".
Understand "Area 24 in Room 4" as a24r4.
the parent of the a24r4 is r4.
the X of the a24r4 is "546080.6892610956".
the Y of the a24r4 is "1753510.9362679888".

[create a25r4]
a25r4 is a area. "An area (25) in r4".
the printed name of the a25r4 is "Room 4".
Understand "Area 25 in Room 4" as a25r4.
the parent of the a25r4 is r4.
the X of the a25r4 is "547596.6892610957".
the Y of the a25r4 is "1753510.9362679888".

[create a29r4]
a29r4 is a area. "An area (29) in r4".
the printed name of the a29r4 is "Room 4".
Understand "Area 29 in Room 4" as a29r4.
the parent of the a29r4 is r4.
the X of the a29r4 is "546093.0873366152".
the Y of the a29r4 is "1753449.159650314".

[create a30r4]
a30r4 is a area. "An area (30) in r4".
the printed name of the a30r4 is "Room 4".
Understand "Area 30 in Room 4" as a30r4.
the parent of the a30r4 is r4.
the X of the a30r4 is "547609.0873366152".
the Y of the a30r4 is "1753449.1596503141".

[create a47r4]
a47r4 is a area. "An area (47) in r4".
the printed name of the a47r4 is "Room 4".
Understand "Area 47 in Room 4" as a47r4.
the parent of the a47r4 is r4.
the X of the a47r4 is "545089.0".
the Y of the a47r4 is "1754514.0".

[create a48r4]
a48r4 is a area. "An area (48) in r4".
the printed name of the a48r4 is "Room 4".
Understand "Area 48 in Room 4" as a48r4.
the parent of the a48r4 is r4.
the X of the a48r4 is "545120.0876068377".
the Y of the a48r4 is "1754488.1662393163".

[create a49r4]
a49r4 is a area. "An area (49) in r4".
the printed name of the a49r4 is "Room 4".
Understand "Area 49 in Room 4" as a49r4.
the parent of the a49r4 is r4.
the X of the a49r4 is "545931.0".
the Y of the a49r4 is "1754514.0".

[create a50r4]
a50r4 is a area. "An area (50) in r4".
the printed name of the a50r4 is "Room 4".
Understand "Area 50 in Room 4" as a50r4.
the parent of the a50r4 is r4.
the X of the a50r4 is "545899.9123931623".
the Y of the a50r4 is "1754488.1662393163".

[create a51r4]
a51r4 is a area. "An area (51) in r4".
the printed name of the a51r4 is "Room 4".
Understand "Area 51 in Room 4" as a51r4.
the parent of the a51r4 is r4.
the X of the a51r4 is "547447.0".
the Y of the a51r4 is "1754514.0".

[create a52r4]
a52r4 is a area. "An area (52) in r4".
the printed name of the a52r4 is "Room 4".
Understand "Area 52 in Room 4" as a52r4.
the parent of the a52r4 is r4.
the X of the a52r4 is "547415.9123931623".
the Y of the a52r4 is "1754488.166239316".

[create a55r4]
a55r4 is a area. "An area (55) in r4".
the printed name of the a55r4 is "Room 4".
Understand "Area 55 in Room 4" as a55r4.
the parent of the a55r4 is r4.
the X of the a55r4 is "546604.0".
the Y of the a55r4 is "1754504.0".

[create a56r4]
a56r4 is a area. "An area (56) in r4".
the printed name of the a56r4 is "Room 4".
Understand "Area 56 in Room 4" as a56r4.
the parent of the a56r4 is r4.
the X of the a56r4 is "546636.9940170939".
the Y of the a56r4 is "1754505.7505982905".

[create a57r4]
a57r4 is a area. "An area (57) in r4".
the printed name of the a57r4 is "Room 4".
Understand "Area 57 in Room 4" as a57r4.
the parent of the a57r4 is r4.
the X of the a57r4 is "548034.0".
the Y of the a57r4 is "1754504.0".

[create a58r4]
a58r4 is a area. "An area (58) in r4".
the printed name of the a58r4 is "Room 4".
Understand "Area 58 in Room 4" as a58r4.
the parent of the a58r4 is r4.
the X of the a58r4 is "548066.994017094".
the Y of the a58r4 is "1754505.7505982905".

[create a59r4]
a59r4 is a area. "An area (59) in r4".
the printed name of the a59r4 is "Room 4".
Understand "Area 59 in Room 4" as a59r4.
the parent of the a59r4 is r4.
the X of the a59r4 is "547071.0".
the Y of the a59r4 is "1754514.0".

[create a60r4]
a60r4 is a area. "An area (60) in r4".
the printed name of the a60r4 is "Room 4".
Understand "Area 60 in Room 4" as a60r4.
the parent of the a60r4 is r4.
the X of the a60r4 is "547039.005982906".
the Y of the a60r4 is "1754505.7505982907".

[create a61r4]
a61r4 is a area. "An area (61) in r4".
the printed name of the a61r4 is "Room 4".
Understand "Area 61 in Room 4" as a61r4.
the parent of the a61r4 is r4.
the X of the a61r4 is "548868.0".
the Y of the a61r4 is "1754424.0".

[create a62r4]
a62r4 is a area. "An area (62) in r4".
the printed name of the a62r4 is "Room 4".
Understand "Area 62 in Room 4" as a62r4.
the parent of the a62r4 is r4.
the X of the a62r4 is "548827.6686930091".
the Y of the a62r4 is "1754442.2965214455".

[create a63r4]
a63r4 is a area. "An area (63) in r4".
the printed name of the a63r4 is "Room 4".
Understand "Area 63 in Room 4" as a63r4.
the parent of the a63r4 is r4.
the X of the a63r4 is "545940.0".
the Y of the a63r4 is "1754424.0".

[create a64r4]
a64r4 is a area. "An area (64) in r4".
the printed name of the a64r4 is "Room 4".
Understand "Area 64 in Room 4" as a64r4.
the parent of the a64r4 is r4.
the X of the a64r4 is "545901.0502336449".
the Y of the a64r4 is "1754442.8300233646".

[create a65r4]
a65r4 is a area. "An area (65) in r4".
the printed name of the a65r4 is "Room 4".
Understand "Area 65 in Room 4" as a65r4.
the parent of the a65r4 is r4.
the X of the a65r4 is "547456.0".
the Y of the a65r4 is "1754424.0".

[create a66r4]
a66r4 is a area. "An area (66) in r4".
the printed name of the a66r4 is "Room 4".
Understand "Area 66 in Room 4" as a66r4.
the parent of the a66r4 is r4.
the X of the a66r4 is "547417.0502336449".
the Y of the a66r4 is "1754442.8300233646".

[create a67r4]
a67r4 is a area. "An area (67) in r4".
the printed name of the a67r4 is "Room 4".
Understand "Area 67 in Room 4" as a67r4.
the parent of the a67r4 is r4.
the X of the a67r4 is "545080.0".
the Y of the a67r4 is "1754424.0".

[create a68r4]
a68r4 is a area. "An area (68) in r4".
the printed name of the a68r4 is "Room 4".
Understand "Area 68 in Room 4" as a68r4.
the parent of the a68r4 is r4.
the X of the a68r4 is "545116.9075342466".
the Y of the a68r4 is "1754443.0342465756".

[create a71r4]
a71r4 is a area. "An area (71) in r4".
the printed name of the a71r4 is "Room 4".
Understand "Area 71 in Room 4" as a71r4.
the parent of the a71r4 is r4.
the X of the a71r4 is "548858.0".
the Y of the a71r4 is "1754514.0".

[create a72r4]
a72r4 is a area. "An area (72) in r4".
the printed name of the a72r4 is "Room 4".
Understand "Area 72 in Room 4" as a72r4.
the parent of the a72r4 is r4.
the X of the a72r4 is "548827.1050372066".
the Y of the a72r4 is "1754487.7894485784".

[create a73r4]
a73r4 is a area. "An area (73) in r4".
the printed name of the a73r4 is "Room 4".
Understand "Area 73 in Room 4" as a73r4.
the parent of the a73r4 is r4.
the X of the a73r4 is "546312.0".
the Y of the a73r4 is "1754514.0".

[create a74r4]
a74r4 is a area. "An area (74) in r4".
the printed name of the a74r4 is "Room 4".
Understand "Area 74 in Room 4" as a74r4.
the parent of the a74r4 is r4.
the X of the a74r4 is "546342.9153780069".
the Y of the a74r4 is "1754488.1834621993".

[create a75r4]
a75r4 is a area. "An area (75) in r4".
the printed name of the a75r4 is "Room 4".
Understand "Area 75 in Room 4" as a75r4.
the parent of the a75r4 is r4.
the X of the a75r4 is "547828.0".
the Y of the a75r4 is "1754514.0".

[create a76r4]
a76r4 is a area. "An area (76) in r4".
the printed name of the a76r4 is "Room 4".
Understand "Area 76 in Room 4" as a76r4.
the parent of the a76r4 is r4.
the X of the a76r4 is "547858.9153780069".
the Y of the a76r4 is "1754488.1834621993".

[create a89r4]
a89r4 is a area. "An area (89) in r4".
the printed name of the a89r4 is "Room 4".
Understand "Area 89 in Room 4" as a89r4.
the parent of the a89r4 is r4.
the X of the a89r4 is "546303.0".
the Y of the a89r4 is "1754424.0".

[create a90r4]
a90r4 is a area. "An area (90) in r4".
the printed name of the a90r4 is "Room 4".
Understand "Area 90 in Room 4" as a90r4.
the parent of the a90r4 is r4.
the X of the a90r4 is "546341.6344476744".
the Y of the a90r4 is "1754442.8615552324".

[create a91r4]
a91r4 is a area. "An area (91) in r4".
the printed name of the a91r4 is "Room 4".
Understand "Area 91 in Room 4" as a91r4.
the parent of the a91r4 is r4.
the X of the a91r4 is "547819.0".
the Y of the a91r4 is "1754424.0".

[create a92r4]
a92r4 is a area. "An area (92) in r4".
the printed name of the a92r4 is "Room 4".
Understand "Area 92 in Room 4" as a92r4.
the parent of the a92r4 is r4.
the X of the a92r4 is "547857.6344476745".
the Y of the a92r4 is "1754442.8615552324".

[create a109r4]
a109r4 is a area. "An area (109) in r4".
the printed name of the a109r4 is "Room 4".
Understand "Area 109 in Room 4" as a109r4.
the parent of the a109r4 is r4.
the X of the a109r4 is "548587.0".
the Y of the a109r4 is "1754514.0".

[create a110r4]
a110r4 is a area. "An area (110) in r4".
the printed name of the a110r4 is "Room 4".
Understand "Area 110 in Room 4" as a110r4.
the parent of the a110r4 is r4.
the X of the a110r4 is "548555.005982906".
the Y of the a110r4 is "1754505.7505982907".

[create a26r5]
a26r5 is a area. "An area (26) in r5".
the printed name of the a26r5 is "Room 5".
Understand "Area 26 in Room 5" as a26r5.
the parent of the a26r5 is r5.
the X of the a26r5 is "546713.8896048588".
the Y of the a26r5 is "1752560.8288873776".

[create a27r5]
a27r5 is a area. "An area (27) in r5".
the printed name of the a27r5 is "Room 5".
Understand "Area 27 in Room 5" as a27r5.
the parent of the a27r5 is r5.
the X of the a27r5 is "546906.1088095646".
the Y of the a27r5 is "1752537.92824688".

[create a28r5]
a28r5 is a area. "An area (28) in r5".
the printed name of the a28r5 is "Room 5".
Understand "Area 28 in Room 5" as a28r5.
the parent of the a28r5 is r5.
the X of the a28r5 is "545394.8633422253".
the Y of the a28r5 is "1752534.3385130367".

[create a34r5]
a34r5 is a area. "An area (34) in r5".
the printed name of the a34r5 is "Room 5".
Understand "Area 34 in Room 5" as a34r5.
the parent of the a34r5 is r5.
the X of the a34r5 is "548229.3355180037".
the Y of the a34r5 is "1752562.3355180037".

[create a35r5]
a35r5 is a area. "An area (35) in r5".
the printed name of the a35r5 is "Room 5".
Understand "Area 35 in Room 5" as a35r5.
the parent of the a35r5 is r5.
the X of the a35r5 is "547851.0670611439".
the Y of the a35r5 is "1752444.7228796843".

[create a36r5]
a36r5 is a area. "An area (36) in r5".
the printed name of the a36r5 is "Room 5".
Understand "Area 36 in Room 5" as a36r5.
the parent of the a36r5 is r5.
the X of the a36r5 is "545743.0171235859".
the Y of the a36r5 is "1752442.915905135".

[create a37r5]
a37r5 is a area. "An area (37) in r5".
the printed name of the a37r5 is "Room 5".
Understand "Area 37 in Room 5" as a37r5.
the parent of the a37r5 is r5.
the X of the a37r5 is "548598.7426126128".
the Y of the a37r5 is "1752441.7067567569".

[create a38r5]
a38r5 is a area. "An area (38) in r5".
the printed name of the a38r5 is "Room 5".
Understand "Area 38 in Room 5" as a38r5.
the parent of the a38r5 is r5.
the X of the a38r5 is "545077.0941579642".
the Y of the a38r5 is "1752441.5825458067".

[create a39r5]
a39r5 is a area. "An area (39) in r5".
the printed name of the a39r5 is "Room 5".
Understand "Area 39 in Room 5" as a39r5.
the parent of the a39r5 is r5.
the X of the a39r5 is "547092.3571189989".
the Y of the a39r5 is "1752456.3797415916".

[create a40r5]
a40r5 is a area. "An area (40) in r5".
the printed name of the a40r5 is "Room 5".
Understand "Area 40 in Room 5" as a40r5.
the parent of the a40r5 is r5.
the X of the a40r5 is "546586.0986075986".
the Y of the a40r5 is "1752477.3255241485".

[create a43r5]
a43r5 is a area. "An area (43) in r5".
the printed name of the a43r5 is "Room 5".
Understand "Area 43 in Room 5" as a43r5.
the parent of the a43r5 is r5.
the X of the a43r5 is "547259.7011034365".
the Y of the a43r5 is "1752443.6497890952".

[create a44r5]
a44r5 is a area. "An area (44) in r5".
the printed name of the a44r5 is "Room 5".
Understand "Area 44 in Room 5" as a44r5.
the parent of the a44r5 is r5.
the X of the a44r5 is "546335.0670611437".
the Y of the a44r5 is "1752444.7228796843".

[create a77r5]
a77r5 is a area. "An area (77) in r5".
the printed name of the a77r5 is "Room 5".
Understand "Area 77 in Room 5" as a77r5.
the parent of the a77r5 is r5.
the X of the a77r5 is "546278.0".
the Y of the a77r5 is "1752480.0".

[create a78r5]
a78r5 is a area. "An area (78) in r5".
the printed name of the a78r5 is "Room 5".
Understand "Area 78 in Room 5" as a78r5.
the parent of the a78r5 is r5.
the X of the a78r5 is "546320.2240682535".
the Y of the a78r5 is "1752461.7923215088".

[create a79r5]
a79r5 is a area. "An area (79) in r5".
the printed name of the a79r5 is "Room 5".
Understand "Area 79 in Room 5" as a79r5.
the parent of the a79r5 is r5.
the X of the a79r5 is "547794.0".
the Y of the a79r5 is "1752480.0".

[create a80r5]
a80r5 is a area. "An area (80) in r5".
the printed name of the a80r5 is "Room 5".
Understand "Area 80 in Room 5" as a80r5.
the parent of the a80r5 is r5.
the X of the a80r5 is "547836.2240682533".
the Y of the a80r5 is "1752461.792321509".

[create a81r5]
a81r5 is a area. "An area (81) in r5".
the printed name of the a81r5 is "Room 5".
Understand "Area 81 in Room 5" as a81r5.
the parent of the a81r5 is r5.
the X of the a81r5 is "547129.8264829654".
the Y of the a81r5 is "1752399.172998213".

[create a82r5]
a82r5 is a area. "An area (82) in r5".
the printed name of the a82r5 is "Room 5".
Understand "Area 82 in Room 5" as a82r5.
the parent of the a82r5 is r5.
the X of the a82r5 is "547096.9390053041".
the Y of the a82r5 is "1752398.3704106302".

[create a83r5]
a83r5 is a area. "An area (83) in r5".
the printed name of the a83r5 is "Room 5".
Understand "Area 83 in Room 5" as a83r5.
the parent of the a83r5 is r5.
the X of the a83r5 is "546547.0".
the Y of the a83r5 is "1752391.0".

[create a84r5]
a84r5 is a area. "An area (84) in r5".
the printed name of the a84r5 is "Room 5".
Understand "Area 84 in Room 5" as a84r5.
the parent of the a84r5 is r5.
the X of the a84r5 is "546578.8743176372".
the Y of the a84r5 is "1752398.7692625208".

[create a93r5]
a93r5 is a area. "An area (93) in r5".
the printed name of the a93r5 is "Room 5".
Understand "Area 93 in Room 5" as a93r5.
the parent of the a93r5 is r5.
the X of the a93r5 is "545790.0".
the Y of the a93r5 is "1752391.0".

[create a94r5]
a94r5 is a area. "An area (94) in r5".
the printed name of the a94r5 is "Room 5".
Understand "Area 94 in Room 5" as a94r5.
the parent of the a94r5 is r5.
the X of the a94r5 is "545758.7862838916".
the Y of the a94r5 is "1752416.633971292".

[create a95r5]
a95r5 is a area. "An area (95) in r5".
the printed name of the a95r5 is "Room 5".
Understand "Area 95 in Room 5" as a95r5.
the parent of the a95r5 is r5.
the X of the a95r5 is "547306.0".
the Y of the a95r5 is "1752391.0".

[create a96r5]
a96r5 is a area. "An area (96) in r5".
the printed name of the a96r5 is "Room 5".
Understand "Area 96 in Room 5" as a96r5.
the parent of the a96r5 is r5.
the X of the a96r5 is "547277.7498557225".
the Y of the a96r5 is "1752416.3342842527".

[create a97r5]
a97r5 is a area. "An area (97) in r5".
the printed name of the a97r5 is "Room 5".
Understand "Area 97 in Room 5" as a97r5.
the parent of the a97r5 is r5.
the X of the a97r5 is "546287.0".
the Y of the a97r5 is "1752391.0".

[create a98r5]
a98r5 is a area. "An area (98) in r5".
the printed name of the a98r5 is "Room 5".
Understand "Area 98 in Room 5" as a98r5.
the parent of the a98r5 is r5.
the X of the a98r5 is "546318.2137161084".
the Y of the a98r5 is "1752416.6339712923".

[create a99r5]
a99r5 is a area. "An area (99) in r5".
the printed name of the a99r5 is "Room 5".
Understand "Area 99 in Room 5" as a99r5.
the parent of the a99r5 is r5.
the X of the a99r5 is "547803.0".
the Y of the a99r5 is "1752391.0".

[create a100r5]
a100r5 is a area. "An area (100) in r5".
the printed name of the a100r5 is "Room 5".
Understand "Area 100 in Room 5" as a100r5.
the parent of the a100r5 is r5.
the X of the a100r5 is "547834.2137161085".
the Y of the a100r5 is "1752416.633971292".

[create a101r5]
a101r5 is a area. "An area (101) in r5".
the printed name of the a101r5 is "Room 5".
Understand "Area 101 in Room 5" as a101r5.
the parent of the a101r5 is r5.
the X of the a101r5 is "545022.0".
the Y of the a101r5 is "1752480.0".

[create a102r5]
a102r5 is a area. "An area (102) in r5".
the printed name of the a102r5 is "Room 5".
Understand "Area 102 in Room 5" as a102r5.
the parent of the a102r5 is r5.
the X of the a102r5 is "545058.0183625785".
the Y of the a102r5 is "1752461.1647782384".

[create a103r5]
a103r5 is a area. "An area (103) in r5".
the printed name of the a103r5 is "Room 5".
Understand "Area 103 in Room 5" as a103r5.
the parent of the a103r5 is r5.
the X of the a103r5 is "548654.0".
the Y of the a103r5 is "1752480.0".

[create a104r5]
a104r5 is a area. "An area (104) in r5".
the printed name of the a104r5 is "Room 5".
Understand "Area 104 in Room 5" as a104r5.
the parent of the a104r5 is r5.
the X of the a104r5 is "548617.1452295461".
the Y of the a104r5 is "1752461.2493588103".

[create a105r5]
a105r5 is a area. "An area (105) in r5".
the printed name of the a105r5 is "Room 5".
Understand "Area 105 in Room 5" as a105r5.
the parent of the a105r5 is r5.
the X of the a105r5 is "545799.0".
the Y of the a105r5 is "1752480.0".

[create a106r5]
a106r5 is a area. "An area (106) in r5".
the printed name of the a106r5 is "Room 5".
Understand "Area 106 in Room 5" as a106r5.
the parent of the a106r5 is r5.
the X of the a106r5 is "545760.2797213208".
the Y of the a106r5 is "1752461.438005709".

[create a107r5]
a107r5 is a area. "An area (107) in r5".
the printed name of the a107r5 is "Room 5".
Understand "Area 107 in Room 5" as a107r5.
the parent of the a107r5 is r5.
the X of the a107r5 is "547315.0".
the Y of the a107r5 is "1752480.0".

[create a108r5]
a108r5 is a area. "An area (108) in r5".
the printed name of the a108r5 is "Room 5".
Understand "Area 108 in Room 5" as a108r5.
the parent of the a108r5 is r5.
the X of the a108r5 is "547276.2797213207".
the Y of the a108r5 is "1752461.438005709".

[create a111r5]
a111r5 is a area. "An area (111) in r5".
the printed name of the a111r5 is "Room 5".
Understand "Area 111 in Room 5" as a111r5.
the parent of the a111r5 is r5.
the X of the a111r5 is "545031.0".
the Y of the a111r5 is "1752391.0".

[create a112r5]
a112r5 is a area. "An area (112) in r5".
the printed name of the a112r5 is "Room 5".
Understand "Area 112 in Room 5" as a112r5.
the parent of the a112r5 is r5.
the X of the a112r5 is "545062.0420577896".
the Y of the a112r5 is "1752416.6166125853".

[create a113r5]
a113r5 is a area. "An area (113) in r5".
the printed name of the a113r5 is "Room 5".
Understand "Area 113 in Room 5" as a113r5.
the parent of the a113r5 is r5.
the X of the a113r5 is "548645.0".
the Y of the a113r5 is "1752391.0".

[create a114r5]
a114r5 is a area. "An area (114) in r5".
the printed name of the a114r5 is "Room 5".
Understand "Area 114 in Room 5" as a114r5.
the parent of the a114r5 is r5.
the X of the a114r5 is "548613.7862838915".
the Y of the a114r5 is "1752416.6339712923".

[create d0]
d0 is a door. "A door between a29r4 and a28r5".
d0 is southwest of a29r4 and northeast of a28r5.

[create d1]
d1 is a door. "A door between a30r4 and a27r5".
d1 is southwest of a30r4 and northeast of a27r5.

[create d2]
d2 is a door. "A door between a29r4 and a26r5".
d2 is southeast of a29r4 and northwest of a26r5.

[create d3]
d3 is a door. "A door between a30r4 and a34r5".
d3 is southeast of a30r4 and northwest of a34r5.

[create d8]
d8 is a door. "A door between a32r2 and a6r3".
d8 is north of a32r2 and south of a6r3.

[create d10]
d10 is a door. "A door between a20r0 and a21r1".
d10 is north of a20r0 and south of a21r1.

[create d12]
d12 is a door. "A door between a22r0 and a14r4".
d12 is north of a22r0 and south of a14r4.

[create d14]
d14 is a door. "A door between a21r1 and a23r2".
d14 is southeast of a21r1 and northwest of a23r2.

east of a0r4 is west of a1r4.

east of a2r4 is west of a3r4.

southeast of a4r4 is northwest of a5r4.

east of a6r3 is west of a7r3.

northwest of a0r4 is southeast of a8r4.

east of a9r4 is west of a10r4.

east of a11r4 is west of a12r4.

west of a11r4 is east of a13r4.

east of a5r4 is west of a14r4.

southeast of a15r1 is northwest of a16r1.

west of a4r4 is east of a17r4.

west of a8r4 is east of a18r4.

southwest of a9r4 is northeast of a14r4.

west of a6r3 is east of a19r3.

southwest of a0r4 is northeast of a24r4.

southwest of a5r4 is northeast of a25r4.

southeast of a11r4 is northwest of a24r4.

southeast of a1r4 is northwest of a25r4.

south of a16r1 is north of a21r1.

east of a26r5 is west of a27r5.

southeast of a20r0 is northwest of a31r0.

southeast of a32r2 is northwest of a33r2.

west of a34r5 is east of a35r5.

east of a28r5 is west of a36r5.

east of a34r5 is west of a37r5.

west of a28r5 is east of a38r5.

southeast of a27r5 is northwest of a39r5.

southwest of a26r5 is northeast of a40r5.

southwest of a22r0 is northeast of a41r0.

southwest of a23r2 is northeast of a42r2.

east of a39r5 is west of a43r5.

west of a40r5 is east of a44r5.

southeast of a45r3 is northwest of a46r3.

southeast of a47r4 is northwest of a48r4.

southwest of a49r4 is northeast of a50r4.

southwest of a51r4 is northeast of a52r4.

southwest of a53r3 is northeast of a54r3.

east of a55r4 is west of a56r4.

east of a57r4 is west of a58r4.

west of a59r4 is east of a60r4.

northwest of a61r4 is southeast of a62r4.

northwest of a63r4 is southeast of a64r4.

northwest of a65r4 is southeast of a66r4.

northeast of a67r4 is southwest of a68r4.

southwest of a69r1 is northeast of a70r1.

southwest of a71r4 is northeast of a72r4.

southeast of a73r4 is northwest of a74r4.

southeast of a75r4 is northwest of a76r4.

southeast of a77r5 is northwest of a78r5.

southeast of a79r5 is northwest of a80r5.

west of a81r5 is east of a82r5.

east of a83r5 is west of a84r5.

northeast of a85r3 is southwest of a86r3.

northwest of a87r3 is southeast of a88r3.

northeast of a89r4 is southwest of a90r4.

northeast of a91r4 is southwest of a92r4.

northwest of a93r5 is southeast of a94r5.

northwest of a95r5 is southeast of a96r5.

northeast of a97r5 is southwest of a98r5.

northeast of a99r5 is southwest of a100r5.

southeast of a101r5 is northwest of a102r5.

southwest of a103r5 is northeast of a104r5.

southwest of a105r5 is northeast of a106r5.

southwest of a107r5 is northeast of a108r5.

west of a109r4 is east of a110r4.

northeast of a111r5 is southwest of a112r5.

northwest of a113r5 is southeast of a114r5.

southwest of a41r0 is northeast of a115r0.

southwest of a42r2 is northeast of a116r2.

southeast of a31r0 is northwest of a117r0.

southeast of a33r2 is northwest of a118r2.

northwest of a15r1 is southeast of a119r1.

northwest of a13r4 is southeast of a48r4.

northeast of a12r4 is southwest of a50r4.

northeast of a3r4 is southwest of a52r4.

northeast of a7r3 is southwest of a54r3.

north of a8r4 is south of a56r4.

north of a4r4 is south of a58r4.

north of a2r4 is south of a60r4.

north of a9r4 is south of a110r4.

southeast of a12r4 is northwest of a64r4.

southeast of a3r4 is northwest of a66r4.

southwest of a13r4 is northeast of a68r4.

northeast of a16r1 is southwest of a70r1.

northeast of a10r4 is southwest of a72r4.

northwest of a18r4 is southeast of a74r4.

northwest of a17r4 is southeast of a76r4.

northwest of a19r3 is southeast of a46r3.

south of a24r4 is north of a29r4.

south of a25r4 is north of a30r4.

northeast of a1r4 is southwest of a2r4.

southwest of a19r3 is northeast of a86r3.

southeast of a7r3 is northwest of a88r3.

southwest of a18r4 is northeast of a90r4.

southwest of a17r4 is northeast of a92r4.

southeast of a10r4 is northwest of a62r4.

northeast of a36r5 is southwest of a106r5.

northeast of a43r5 is southwest of a108r5.

northwest of a44r5 is southeast of a78r5.

northwest of a35r5 is southeast of a80r5.

south of a39r5 is north of a82r5.

south of a40r5 is north of a84r5.

east of a23r2 is west of a32r2.

west of a20r0 is east of a22r0.

southwest of a38r5 is northeast of a112r5.

southeast of a37r5 is northwest of a114r5.

southeast of a36r5 is northwest of a94r5.

southeast of a43r5 is northwest of a96r5.

southwest of a44r5 is northeast of a98r5.

southwest of a35r5 is northeast of a100r5.

northwest of a38r5 is southeast of a102r5.

northeast of a37r5 is southwest of a104r5.

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
the player is in a90r4.

the orientation of the player is 6.
