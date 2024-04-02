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

[create a0r0]
a0r0 is a area. "An area (0) in r0".
the printed name of the a0r0 is "Room 0".
Understand "Area 0 in Room 0" as a0r0.
the parent of the a0r0 is r0.
the X of the a0r0 is "548687.0".
the Y of the a0r0 is "1184514.0".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "548655.9123931623".
the Y of the a1r0 is "1184488.1662393163".

[create a2r0]
a2r0 is a area. "An area (2) in r0".
the printed name of the a2r0 is "Room 0".
Understand "Area 2 in Room 0" as a2r0.
the parent of the a2r0 is r0.
the X of the a2r0 is "549827.0".
the Y of the a2r0 is "1184514.0".

[create a3r0]
a3r0 is a area. "An area (3) in r0".
the printed name of the a3r0 is "Room 0".
Understand "Area 3 in Room 0" as a3r0.
the parent of the a3r0 is r0.
the X of the a3r0 is "549795.9123931623".
the Y of the a3r0 is "1184488.1662393163".

[create a4r0]
a4r0 is a area. "An area (4) in r0".
the printed name of the a4r0 is "Room 0".
Understand "Area 4 in Room 0" as a4r0.
the parent of the a4r0 is r0.
the X of the a4r0 is "546199.0".
the Y of the a4r0 is "1184504.0".

[create a5r0]
a5r0 is a area. "An area (5) in r0".
the printed name of the a5r0 is "Room 0".
Understand "Area 5 in Room 0" as a5r0.
the parent of the a5r0 is r0.
the X of the a5r0 is "546230.4469420139".
the Y of the a5r0 is "1184505.9053057984".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "547844.0".
the Y of the a6r0 is "1184504.0".

[create a7r0]
a7r0 is a area. "An area (7) in r0".
the printed name of the a7r0 is "Room 0".
Understand "Area 7 in Room 0" as a7r0.
the parent of the a7r0 is r0.
the X of the a7r0 is "547876.9940170939".
the Y of the a7r0 is "1184505.7505982905".

[create a8r0]
a8r0 is a area. "An area (8) in r0".
the printed name of the a8r0 is "Room 0".
Understand "Area 8 in Room 0" as a8r0.
the parent of the a8r0 is r0.
the X of the a8r0 is "548428.0".
the Y of the a8r0 is "1184504.0".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "548395.005982906".
the Y of the a9r0 is "1184505.7505982905".

[create a10r0]
a10r0 is a area. "An area (10) in r0".
the printed name of the a10r0 is "Room 0".
Understand "Area 10 in Room 0" as a10r0.
the parent of the a10r0 is r0.
the X of the a10r0 is "546631.0".
the Y of the a10r0 is "1184523.0".

[create a11r0]
a11r0 is a area. "An area (11) in r0".
the printed name of the a11r0 is "Room 0".
Understand "Area 11 in Room 0" as a11r0.
the parent of the a11r0 is r0.
the X of the a11r0 is "546600.7540392072".
the Y of the a11r0 is "1184513.5".

[create a12r0]
a12r0 is a area. "An area (12) in r0".
the printed name of the a12r0 is "Room 0".
Understand "Area 12 in Room 0" as a12r0.
the parent of the a12r0 is r0.
the X of the a12r0 is "547607.0".
the Y of the a12r0 is "1184424.0".

[create a13r0]
a13r0 is a area. "An area (13) in r0".
the printed name of the a13r0 is "Room 0".
Understand "Area 13 in Room 0" as a13r0.
the parent of the a13r0 is r0.
the X of the a13r0 is "547649.06".
the Y of the a13r0 is "1184442.5189999999".

[create a14r0]
a14r0 is a area. "An area (14) in r0".
the printed name of the a14r0 is "Room 0".
Understand "Area 14 in Room 0" as a14r0.
the parent of the a14r0 is r0.
the X of the a14r0 is "549059.0".
the Y of the a14r0 is "1184424.0".

[create a15r0]
a15r0 is a area. "An area (15) in r0".
the printed name of the a15r0 is "Room 0".
Understand "Area 15 in Room 0" as a15r0.
the parent of the a15r0 is r0.
the X of the a15r0 is "549097.6344476745".
the Y of the a15r0 is "1184442.8615552324".

[create a16r0]
a16r0 is a area. "An area (16) in r0".
the printed name of the a16r0 is "Room 0".
Understand "Area 16 in Room 0" as a16r0.
the parent of the a16r0 is r0.
the X of the a16r0 is "548696.0".
the Y of the a16r0 is "1184424.0".

[create a17r0]
a17r0 is a area. "An area (17) in r0".
the printed name of the a17r0 is "Room 0".
Understand "Area 17 in Room 0" as a17r0.
the parent of the a17r0 is r0.
the X of the a17r0 is "548657.0502336449".
the Y of the a17r0 is "1184442.8300233644".

[create a18r0]
a18r0 is a area. "An area (18) in r0".
the printed name of the a18r0 is "Room 0".
Understand "Area 18 in Room 0" as a18r0.
the parent of the a18r0 is r0.
the X of the a18r0 is "549836.0".
the Y of the a18r0 is "1184424.0".

[create a19r0]
a19r0 is a area. "An area (19) in r0".
the printed name of the a19r0 is "Room 0".
Understand "Area 19 in Room 0" as a19r0.
the parent of the a19r0 is r0.
the X of the a19r0 is "549799.7196261683".
the Y of the a19r0 is "1184443.096962617".

[create a20r0]
a20r0 is a area. "An area (20) in r0".
the printed name of the a20r0 is "Room 0".
Understand "Area 20 in Room 0" as a20r0.
the parent of the a20r0 is r0.
the X of the a20r0 is "545117.0".
the Y of the a20r0 is "1184514.0".

[create a21r0]
a21r0 is a area. "An area (21) in r0".
the printed name of the a21r0 is "Room 0".
Understand "Area 21 in Room 0" as a21r0.
the parent of the a21r0 is r0.
the X of the a21r0 is "545147.9153780066".
the Y of the a21r0 is "1184488.1834621993".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "549068.0".
the Y of the a22r0 is "1184514.0".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "549098.915378007".
the Y of the a23r0 is "1184488.1834621993".

[create a24r0]
a24r0 is a area. "An area (24) in r0".
the printed name of the a24r0 is "Room 0".
Understand "Area 24 in Room 0" as a24r0.
the parent of the a24r0 is r0.
the X of the a24r0 is "547011.0".
the Y of the a24r0 is "1184514.0".

[create a25r0]
a25r0 is a area. "An area (25) in r0".
the printed name of the a25r0 is "Room 0".
Understand "Area 25 in Room 0" as a25r0.
the parent of the a25r0 is r0.
the X of the a25r0 is "546979.9123931624".
the Y of the a25r0 is "1184488.1662393163".

[create a26r0]
a26r0 is a area. "An area (26) in r0".
the printed name of the a26r0 is "Room 0".
Understand "Area 26 in Room 0" as a26r0.
the parent of the a26r0 is r0.
the X of the a26r0 is "547616.0".
the Y of the a26r0 is "1184514.0".

[create a27r0]
a27r0 is a area. "An area (27) in r0".
the printed name of the a27r0 is "Room 0".
Understand "Area 27 in Room 0" as a27r0.
the parent of the a27r0 is r0.
the X of the a27r0 is "547647.0876068377".
the Y of the a27r0 is "1184488.1662393163".

[create a36r0]
a36r0 is a area. "An area (36) in r0".
the printed name of the a36r0 is "Room 0".
Understand "Area 36 in Room 0" as a36r0.
the parent of the a36r0 is r0.
the X of the a36r0 is "546392.0".
the Y of the a36r0 is "1183810.0".

[create a37r0]
a37r0 is a area. "An area (37) in r0".
the printed name of the a37r0 is "Room 0".
Understand "Area 37 in Room 0" as a37r0.
the parent of the a37r0 is r0.
the X of the a37r0 is "546361.7769924193".
the Y of the a37r0 is "1183783.9929317762".

[create a38r0]
a38r0 is a area. "An area (38) in r0".
the printed name of the a38r0 is "Room 0".
Understand "Area 38 in Room 0" as a38r0.
the parent of the a38r0 is r0.
the X of the a38r0 is "546284.0".
the Y of the a38r0 is "1183822.0".

[create a39r0]
a39r0 is a area. "An area (39) in r0".
the printed name of the a39r0 is "Room 0".
Understand "Area 39 in Room 0" as a39r0.
the parent of the a39r0 is r0.
the X of the a39r0 is "546307.7621979022".
the Y of the a39r0 is "1183789.859781122".

[create a40r0]
a40r0 is a area. "An area (40) in r0".
the printed name of the a40r0 is "Room 0".
Understand "Area 40 in Room 0" as a40r0.
the parent of the a40r0 is r0.
the X of the a40r0 is "545108.0".
the Y of the a40r0 is "1184424.0".

[create a41r0]
a41r0 is a area. "An area (41) in r0".
the printed name of the a41r0 is "Room 0".
Understand "Area 41 in Room 0" as a41r0.
the parent of the a41r0 is r0.
the X of the a41r0 is "545150.06".
the Y of the a41r0 is "1184442.5189999999".

[create a42r0]
a42r0 is a area. "An area (42) in r0".
the printed name of the a42r0 is "Room 0".
Understand "Area 42 in Room 0" as a42r0.
the parent of the a42r0 is r0.
the X of the a42r0 is "547020.0".
the Y of the a42r0 is "1184424.0".

[create a43r0]
a43r0 is a area. "An area (43) in r0".
the printed name of the a43r0 is "Room 0".
Understand "Area 43 in Room 0" as a43r0.
the parent of the a43r0 is r0.
the X of the a43r0 is "546977.94".
the Y of the a43r0 is "1184442.519".

[create a78r0]
a78r0 is a area. "An area (78) in r0".
the printed name of the a78r0 is "Room 0".
Understand "Area 78 in Room 0" as a78r0.
the parent of the a78r0 is r0.
the X of the a78r0 is "549780.5896029256".
the Y of the a78r0 is "1184462.7304075237".

[create a79r0]
a79r0 is a area. "An area (79) in r0".
the printed name of the a79r0 is "Room 0".
Understand "Area 79 in Room 0" as a79r0.
the parent of the a79r0 is r0.
the X of the a79r0 is "546566.5035140851".
the Y of the a79r0 is "1184366.6109870176".

[create a80r0]
a80r0 is a area. "An area (80) in r0".
the printed name of the a80r0 is "Room 0".
Understand "Area 80 in Room 0" as a80r0.
the parent of the a80r0 is r0.
the X of the a80r0 is "546603.5370678175".
the Y of the a80r0 is "1184335.9862183537".

[create a81r0]
a81r0 is a area. "An area (81) in r0".
the printed name of the a81r0 is "Room 0".
Understand "Area 81 in Room 0" as a81r0.
the parent of the a81r0 is r0.
the X of the a81r0 is "545165.092671901".
the Y of the a81r0 is "1184459.8065726862".

[create a82r0]
a82r0 is a area. "An area (82) in r0".
the printed name of the a82r0 is "Room 0".
Understand "Area 82 in Room 0" as a82r0.
the parent of the a82r0 is r0.
the X of the a82r0 is "549114.958925213".
the Y of the a82r0 is "1184461.6795222145".

[create a83r0]
a83r0 is a area. "An area (83) in r0".
the printed name of the a83r0 is "Room 0".
Understand "Area 83 in Room 0" as a83r0.
the parent of the a83r0 is r0.
the X of the a83r0 is "546962.8606168444".
the Y of the a83r0 is "1184459.860290629".

[create a84r0]
a84r0 is a area. "An area (84) in r0".
the printed name of the a84r0 is "Room 0".
Understand "Area 84 in Room 0" as a84r0.
the parent of the a84r0 is r0.
the X of the a84r0 is "547664.1393831554".
the Y of the a84r0 is "1184459.8602906289".

[create a85r0]
a85r0 is a area. "An area (85) in r0".
the printed name of the a85r0 is "Room 0".
Understand "Area 85 in Room 0" as a85r0.
the parent of the a85r0 is r0.
the X of the a85r0 is "548639.9343690765".
the Y of the a85r0 is "1184461.6427193342".

[create a86r0]
a86r0 is a area. "An area (86) in r0".
the printed name of the a86r0 is "Room 0".
Understand "Area 86 in Room 0" as a86r0.
the parent of the a86r0 is r0.
the X of the a86r0 is "546237.4358112005".
the Y of the a86r0 is "1184441.694296065".

[create a87r0]
a87r0 is a area. "An area (87) in r0".
the printed name of the a87r0 is "Room 0".
Understand "Area 87 in Room 0" as a87r0.
the parent of the a87r0 is r0.
the X of the a87r0 is "548836.6892610956".
the Y of the a87r0 is "1183510.936267989".

[create a89r0]
a89r0 is a area. "An area (89) in r0".
the printed name of the a89r0 is "Room 0".
Understand "Area 89 in Room 0" as a89r0.
the parent of the a89r0 is r0.
the X of the a89r0 is "546328.858357395".
the Y of the a89r0 is "1183760.983477703".

[create a98r0]
a98r0 is a area. "An area (98) in r0".
the printed name of the a98r0 is "Room 0".
Understand "Area 98 in Room 0" as a98r0.
the parent of the a98r0 is r0.
the X of the a98r0 is "547884.8774585926".
the Y of the a98r0 is "1184428.098699534".

[create a99r0]
a99r0 is a area. "An area (99) in r0".
the printed name of the a99r0 is "Room 0".
Understand "Area 99 in Room 0" as a99r0.
the parent of the a99r0 is r0.
the X of the a99r0 is "548388.2709163344".
the Y of the a99r0 is "1184439.410192563".

[create a103r0]
a103r0 is a area. "An area (103) in r0".
the printed name of the a103r0 is "Room 0".
Understand "Area 103 in Room 0" as a103r0.
the parent of the a103r0 is r0.
the X of the a103r0 is "549464.1625855833".
the Y of the a103r0 is "1184370.32035392".

[create a104r0]
a104r0 is a area. "An area (104) in r0".
the printed name of the a104r0 is "Room 0".
Understand "Area 104 in Room 0" as a104r0.
the parent of the a104r0 is r0.
the X of the a104r0 is "545524.9141586567".
the Y of the a104r0 is "1184333.2035531262".

[create a105r0]
a105r0 is a area. "An area (105) in r0".
the printed name of the a105r0 is "Room 0".
Understand "Area 105 in Room 0" as a105r0.
the parent of the a105r0 is r0.
the X of the a105r0 is "548253.8466789336".
the Y of the a105r0 is "1184334.8466789338".

[create a106r0]
a106r0 is a area. "An area (106) in r0".
the printed name of the a106r0 is "Room 0".
Understand "Area 106 in Room 0" as a106r0.
the parent of the a106r0 is r0.
the X of the a106r0 is "548041.6779207943".
the Y of the a106r0 is "1184319.3111993254".

[create a107r0]
a107r0 is a area. "An area (107) in r0".
the printed name of the a107r0 is "Room 0".
Understand "Area 107 in Room 0" as a107r0.
the parent of the a107r0 is r0.
the X of the a107r0 is "546259.7255".
the Y of the a107r0 is "1183466.0".

[create a108r0]
a108r0 is a area. "An area (108) in r0".
the printed name of the a108r0 is "Room 0".
Understand "Area 108 in Room 0" as a108r0.
the parent of the a108r0 is r0.
the X of the a108r0 is "548008.46433213".
the Y of the a108r0 is "1183494.0".

[create a109r0]
a109r0 is a area. "An area (109) in r0".
the printed name of the a109r0 is "Room 0".
Understand "Area 109 in Room 0" as a109r0.
the parent of the a109r0 is r0.
the X of the a109r0 is "545526.8626086956".
the Y of the a109r0 is "1183466.0".

[create a110r0]
a110r0 is a area. "An area (110) in r0".
the printed name of the a110r0 is "Room 0".
Understand "Area 110 in Room 0" as a110r0.
the parent of the a110r0 is r0.
the X of the a110r0 is "546618.7785379061".
the Y of the a110r0 is "1183494.0".

[create a28r1]
a28r1 is a area. "An area (28) in r1".
the printed name of the a28r1 is "Room 1".
Understand "Area 28 in Room 1" as a28r1.
the parent of the a28r1 is r1.
the X of the a28r1 is "548168.9654544938".
the Y of the a28r1 is "1182770.3608086207".

[create a29r1]
a29r1 is a area. "An area (29) in r1".
the printed name of the a29r1 is "Room 1".
Understand "Area 29 in Room 1" as a29r1.
the parent of the a29r1 is r1.
the X of the a29r1 is "548192.207201147".
the Y of the a29r1 is "1182796.210478726".

[create a30r1]
a30r1 is a area. "An area (30) in r1".
the printed name of the a30r1 is "Room 1".
Understand "Area 30 in Room 1" as a30r1.
the parent of the a30r1 is r1.
the X of the a30r1 is "546744.7528572587".
the Y of the a30r1 is "1182831.9315068494".

[create a31r1]
a31r1 is a area. "An area (31) in r1".
the printed name of the a31r1 is "Room 1".
Understand "Area 31 in Room 1" as a31r1.
the parent of the a31r1 is r1.
the X of the a31r1 is "546740.985170596".
the Y of the a31r1 is "1182864.161133258".

[create a32r1]
a32r1 is a area. "An area (32) in r1".
the printed name of the a32r1 is "Room 1".
Understand "Area 32 in Room 1" as a32r1.
the parent of the a32r1 is r1.
the X of the a32r1 is "546284.0".
the Y of the a32r1 is "1183110.0".

[create a33r1]
a33r1 is a area. "An area (33) in r1".
the printed name of the a33r1 is "Room 1".
Understand "Area 33 in Room 1" as a33r1.
the parent of the a33r1 is r1.
the X of the a33r1 is "546307.0755088733".
the Y of the a33r1 is "1183142.0063361113".

[create a34r1]
a34r1 is a area. "An area (34) in r1".
the printed name of the a34r1 is "Room 1".
Understand "Area 34 in Room 1" as a34r1.
the parent of the a34r1 is r1.
the X of the a34r1 is "546392.0".
the Y of the a34r1 is "1183125.0".

[create a35r1]
a35r1 is a area. "An area (35) in r1".
the printed name of the a35r1 is "Room 1".
Understand "Area 35 in Room 1" as a35r1.
the parent of the a35r1 is r1.
the X of the a35r1 is "546360.8077975204".
the Y of the a35r1 is "1183151.4338578528".

[create a44r1]
a44r1 is a area. "An area (44) in r1".
the printed name of the a44r1 is "Room 1".
Understand "Area 44 in Room 1" as a44r1.
the parent of the a44r1 is r1.
the X of the a44r1 is "547020.0".
the Y of the a44r1 is "1182391.0".

[create a45r1]
a45r1 is a area. "An area (45) in r1".
the printed name of the a45r1 is "Room 1".
Understand "Area 45 in Room 1" as a45r1.
the parent of the a45r1 is r1.
the X of the a45r1 is "546988.7862838916".
the Y of the a45r1 is "1182416.633971292".

[create a46r1]
a46r1 is a area. "An area (46) in r1".
the printed name of the a46r1 is "Room 1".
Understand "Area 46 in Room 1" as a46r1.
the parent of the a46r1 is r1.
the X of the a46r1 is "549894.0".
the Y of the a46r1 is "1182480.0".

[create a47r1]
a47r1 is a area. "An area (47) in r1".
the printed name of the a47r1 is "Room 1".
Understand "Area 47 in Room 1" as a47r1.
the parent of the a47r1 is r1.
the X of the a47r1 is "549857.1452295461".
the Y of the a47r1 is "1182461.24935881".

[create a48r1]
a48r1 is a area. "An area (48) in r1".
the printed name of the a48r1 is "Room 1".
Understand "Area 48 in Room 1" as a48r1.
the parent of the a48r1 is r1.
the X of the a48r1 is "548555.0".
the Y of the a48r1 is "1182480.0".

[create a49r1]
a49r1 is a area. "An area (49) in r1".
the printed name of the a49r1 is "Room 1".
Understand "Area 49 in Room 1" as a49r1.
the parent of the a49r1 is r1.
the X of the a49r1 is "548516.2797213211".
the Y of the a49r1 is "1182461.4380057093".

[create a50r1]
a50r1 is a area. "An area (50) in r1".
the printed name of the a50r1 is "Room 1".
Understand "Area 50 in Room 1" as a50r1.
the parent of the a50r1 is r1.
the X of the a50r1 is "549034.0".
the Y of the a50r1 is "1182480.0".

[create a51r1]
a51r1 is a area. "An area (51) in r1".
the printed name of the a51r1 is "Room 1".
Understand "Area 51 in Room 1" as a51r1.
the parent of the a51r1 is r1.
the X of the a51r1 is "549076.2240682533".
the Y of the a51r1 is "1182461.792321509".

[create a52r1]
a52r1 is a area. "An area (52) in r1".
the printed name of the a52r1 is "Room 1".
Understand "Area 52 in Room 1" as a52r1.
the parent of the a52r1 is r1.
the X of the a52r1 is "547029.0".
the Y of the a52r1 is "1182480.0".

[create a53r1]
a53r1 is a area. "An area (53) in r1".
the printed name of the a53r1 is "Room 1".
Understand "Area 53 in Room 1" as a53r1.
the parent of the a53r1 is r1.
the X of the a53r1 is "546986.1485252036".
the Y of the a53r1 is "1182461.8557671143".

[create a54r1]
a54r1 is a area. "An area (54) in r1".
the printed name of the a54r1 is "Room 1".
Understand "Area 54 in Room 1" as a54r1.
the parent of the a54r1 is r1.
the X of the a54r1 is "548436.5136334812".
the Y of the a54r1 is "1182396.0792644261".

[create a55r1]
a55r1 is a area. "An area (55) in r1".
the printed name of the a55r1 is "Room 1".
Understand "Area 55 in Room 1" as a55r1.
the parent of the a55r1 is r1.
the X of the a55r1 is "548404.0337648757".
the Y of the a55r1 is "1182396.7981430758".

[create a56r1]
a56r1 is a area. "An area (56) in r1".
the printed name of the a56r1 is "Room 1".
Understand "Area 56 in Room 1" as a56r1.
the parent of the a56r1 is r1.
the X of the a56r1 is "546177.0898876404".
the Y of the a56r1 is "1182400.0".

[create a57r1]
a57r1 is a area. "An area (57) in r1".
the printed name of the a57r1 is "Room 1".
Understand "Area 57 in Room 1" as a57r1.
the parent of the a57r1 is r1.
the X of the a57r1 is "546210.0417059883".
the Y of the a57r1 is "1182398.786189433".

[create a58r1]
a58r1 is a area. "An area (58) in r1".
the printed name of the a58r1 is "Room 1".
Understand "Area 58 in Room 1" as a58r1.
the parent of the a58r1 is r1.
the X of the a58r1 is "547786.0988067101".
the Y of the a58r1 is "1182399.9118003112".

[create a59r1]
a59r1 is a area. "An area (59) in r1".
the printed name of the a59r1 is "Room 1".
Understand "Area 59 in Room 1" as a59r1.
the parent of the a59r1 is r1.
the X of the a59r1 is "547818.8763853854".
the Y of the a59r1 is "1182398.7249208104".

[create a60r1]
a60r1 is a area. "An area (60) in r1".
the printed name of the a60r1 is "Room 1".
Understand "Area 60 in Room 1" as a60r1.
the parent of the a60r1 is r1.
the X of the a60r1 is "547594.0".
the Y of the a60r1 is "1182391.0".

[create a61r1]
a61r1 is a area. "An area (61) in r1".
the printed name of the a61r1 is "Room 1".
Understand "Area 61 in Room 1" as a61r1.
the parent of the a61r1 is r1.
the X of the a61r1 is "547624.8976513586".
the Y of the a61r1 is "1182416.602009688".

[create a62r1]
a62r1 is a area. "An area (62) in r1".
the printed name of the a62r1 is "Room 1".
Understand "Area 62 in Room 1" as a62r1.
the parent of the a62r1 is r1.
the X of the a62r1 is "545092.0".
the Y of the a62r1 is "1182391.0".

[create a63r1]
a63r1 is a area. "An area (63) in r1".
the printed name of the a63r1 is "Room 1".
Understand "Area 63 in Room 1" as a63r1.
the parent of the a63r1 is r1.
the X of the a63r1 is "545123.2137161086".
the Y of the a63r1 is "1182416.6339712918".

[create a64r1]
a64r1 is a area. "An area (64) in r1".
the printed name of the a64r1 is "Room 1".
Understand "Area 64 in Room 1" as a64r1.
the parent of the a64r1 is r1.
the X of the a64r1 is "549885.0".
the Y of the a64r1 is "1182391.0".

[create a65r1]
a65r1 is a area. "An area (65) in r1".
the printed name of the a65r1 is "Room 1".
Understand "Area 65 in Room 1" as a65r1.
the parent of the a65r1 is r1.
the X of the a65r1 is "549853.7862838916".
the Y of the a65r1 is "1182416.6339712918".

[create a66r1]
a66r1 is a area. "An area (66) in r1".
the printed name of the a66r1 is "Room 1".
Understand "Area 66 in Room 1" as a66r1.
the parent of the a66r1 is r1.
the X of the a66r1 is "548546.0".
the Y of the a66r1 is "1182391.0".

[create a67r1]
a67r1 is a area. "An area (67) in r1".
the printed name of the a67r1 is "Room 1".
Understand "Area 67 in Room 1" as a67r1.
the parent of the a67r1 is r1.
the X of the a67r1 is "548519.7354860334".
the Y of the a67r1 is "1182416.1334901764".

[create a68r1]
a68r1 is a area. "An area (68) in r1".
the printed name of the a68r1 is "Room 1".
Understand "Area 68 in Room 1" as a68r1.
the parent of the a68r1 is r1.
the X of the a68r1 is "545083.0".
the Y of the a68r1 is "1182480.0".

[create a69r1]
a69r1 is a area. "An area (69) in r1".
the printed name of the a69r1 is "Room 1".
Understand "Area 69 in Room 1" as a69r1.
the parent of the a69r1 is r1.
the X of the a69r1 is "545114.0059757694".
the Y of the a69r1 is "1182460.657907662".

[create a70r1]
a70r1 is a area. "An area (70) in r1".
the printed name of the a70r1 is "Room 1".
Understand "Area 70 in Room 1" as a70r1.
the parent of the a70r1 is r1.
the X of the a70r1 is "547585.0".
the Y of the a70r1 is "1182480.0".

[create a71r1]
a71r1 is a area. "An area (71) in r1".
the printed name of the a71r1 is "Room 1".
Understand "Area 71 in Room 1" as a71r1.
the parent of the a71r1 is r1.
the X of the a71r1 is "547616.0059757694".
the Y of the a71r1 is "1182460.6579076622".

[create a72r1]
a72r1 is a area. "An area (72) in r1".
the printed name of the a72r1 is "Room 1".
Understand "Area 72 in Room 1" as a72r1.
the parent of the a72r1 is r1.
the X of the a72r1 is "549043.0".
the Y of the a72r1 is "1182391.0".

[create a73r1]
a73r1 is a area. "An area (73) in r1".
the printed name of the a73r1 is "Room 1".
Understand "Area 73 in Room 1" as a73r1.
the parent of the a73r1 is r1.
the X of the a73r1 is "549074.2137161086".
the Y of the a73r1 is "1182416.633971292".

[create a74r1]
a74r1 is a area. "An area (74) in r1".
the printed name of the a74r1 is "Room 1".
Understand "Area 74 in Room 1" as a74r1.
the parent of the a74r1 is r1.
the X of the a74r1 is "547639.2874869031".
the Y of the a74r1 is "1182440.3485043326".

[create a75r1]
a75r1 is a area. "An area (75) in r1".
the printed name of the a75r1 is "Room 1".
Understand "Area 75 in Room 1" as a75r1.
the parent of the a75r1 is r1.
the X of the a75r1 is "545137.389199255".
the Y of the a75r1 is "1182440.2597765361".

[create a76r1]
a76r1 is a area. "An area (76) in r1".
the printed name of the a76r1 is "Room 1".
Understand "Area 76 in Room 1" as a76r1.
the parent of the a76r1 is r1.
the X of the a76r1 is "549838.7426126127".
the Y of the a76r1 is "1182441.7067567566".

[create a77r1]
a77r1 is a area. "An area (77) in r1".
the printed name of the a77r1 is "Room 1".
Understand "Area 77 in Room 1" as a77r1.
the parent of the a77r1 is r1.
the X of the a77r1 is "548500.0109539658".
the Y of the a77r1 is "1182443.9822467442".

[create a88r1]
a88r1 is a area. "An area (88) in r1".
the printed name of the a88r1 is "Room 1".
Understand "Area 88 in Room 1" as a88r1.
the parent of the a88r1 is r1.
the X of the a88r1 is "548849.0873366152".
the Y of the a88r1 is "1183449.1596503141".

[create a90r1]
a90r1 is a area. "An area (90) in r1".
the printed name of the a90r1 is "Room 1".
Understand "Area 90 in Room 1" as a90r1.
the parent of the a90r1 is r1.
the X of the a90r1 is "546971.7817693059".
the Y of the a90r1 is "1182444.9748289345".

[create a91r1]
a91r1 is a area. "An area (91) in r1".
the printed name of the a91r1 is "Room 1".
Understand "Area 91 in Room 1" as a91r1.
the parent of the a91r1 is r1.
the X of the a91r1 is "548402.3882293317".
the Y of the a91r1 is "1182448.118209669".

[create a92r1]
a92r1 is a area. "An area (92) in r1".
the printed name of the a92r1 is "Room 1".
Understand "Area 92 in Room 1" as a92r1.
the parent of the a92r1 is r1.
the X of the a92r1 is "546215.0549596348".
the Y of the a92r1 is "1182453.5783605264".

[create a93r1]
a93r1 is a area. "An area (93) in r1".
the printed name of the a93r1 is "Room 1".
Understand "Area 93 in Room 1" as a93r1.
the parent of the a93r1 is r1.
the X of the a93r1 is "547824.8756533298".
the Y of the a93r1 is "1182464.9246203678".

[create a94r1]
a94r1 is a area. "An area (94) in r1".
the printed name of the a94r1 is "Room 1".
Understand "Area 94 in Room 1" as a94r1.
the parent of the a94r1 is r1.
the X of the a94r1 is "548263.1925586006".
the Y of the a94r1 is "1182738.0684377798".

[create a95r1]
a95r1 is a area. "An area (95) in r1".
the printed name of the a95r1 is "Room 1".
Understand "Area 95 in Room 1" as a95r1.
the parent of the a95r1 is r1.
the X of the a95r1 is "546607.5654716168".
the Y of the a95r1 is "1182862.604757033".

[create a96r1]
a96r1 is a area. "An area (96) in r1".
the printed name of the a96r1 is "Room 1".
Understand "Area 96 in Room 1" as a96r1.
the parent of the a96r1 is r1.
the X of the a96r1 is "546328.9241359931".
the Y of the a96r1 is "1183172.8985341636".

[create a97r1]
a97r1 is a area. "An area (97) in r1".
the printed name of the a97r1 is "Room 1".
Understand "Area 97 in Room 1" as a97r1.
the parent of the a97r1 is r1.
the X of the a97r1 is "549091.0670611439".
the Y of the a97r1 is "1182444.7228796843".

[create a100r1]
a100r1 is a area. "An area (100) in r1".
the printed name of the a100r1 is "Room 1".
Understand "Area 100 in Room 1" as a100r1.
the parent of the a100r1 is r1.
the X of the a100r1 is "546606.025639172".
the Y of the a100r1 is "1182566.8975919492".

[create a101r1]
a101r1 is a area. "An area (101) in r1".
the printed name of the a101r1 is "Room 1".
Understand "Area 101 in Room 1" as a101r1.
the parent of the a101r1 is r1.
the X of the a101r1 is "545524.6666860587".
the Y of the a101r1 is "1182572.9466292777".

[create a102r1]
a102r1 is a area. "An area (102) in r1".
the printed name of the a102r1 is "Room 1".
Understand "Area 102 in Room 1" as a102r1.
the parent of the a102r1 is r1.
the X of the a102r1 is "549469.3355180037".
the Y of the a102r1 is "1182562.3355180037".

[create a111r1]
a111r1 is a area. "An area (111) in r1".
the printed name of the a111r1 is "Room 1".
Understand "Area 111 in Room 1" as a111r1.
the parent of the a111r1 is r1.
the X of the a111r1 is "548093.8228486197".
the Y of the a111r1 is "1182592.3813950908".

[create a112r1]
a112r1 is a area. "An area (112) in r1".
the printed name of the a112r1 is "Room 1".
Understand "Area 112 in Room 1" as a112r1.
the parent of the a112r1 is r1.
the X of the a112r1 is "548060.1066303768".
the Y of the a112r1 is "1182600.9003866643".

[create d0]
d0 is a door. "A door between a87r0 and a88r1".
d0 is south of a87r0 and north of a88r1.

[create d1]
d1 is a door. "A door between a108r0 and a112r1".
d1 is south of a108r0 and north of a112r1.

[create d2]
d2 is a door. "A door between a110r0 and a95r1".
d2 is south of a110r0 and north of a95r1.

[create d3]
d3 is a door. "A door between a107r0 and a96r1".
d3 is south of a107r0 and north of a96r1.

[create d4]
d4 is a door. "A door between a109r0 and a101r1".
d4 is south of a109r0 and north of a101r1.

southwest of a0r0 is northeast of a1r0.

southwest of a2r0 is northeast of a3r0.

east of a4r0 is west of a5r0.

east of a6r0 is west of a7r0.

west of a8r0 is east of a9r0.

west of a10r0 is east of a11r0.

northeast of a12r0 is southwest of a13r0.

northeast of a14r0 is southwest of a15r0.

northwest of a16r0 is southeast of a17r0.

northwest of a18r0 is southeast of a19r0.

southeast of a20r0 is northwest of a21r0.

southeast of a22r0 is northwest of a23r0.

southwest of a24r0 is northeast of a25r0.

southeast of a26r0 is northwest of a27r0.

northeast of a28r1 is southwest of a29r1.

north of a30r1 is south of a31r1.

northeast of a32r1 is southwest of a33r1.

northwest of a34r1 is southeast of a35r1.

southwest of a36r0 is northeast of a37r0.

southeast of a38r0 is northwest of a39r0.

northeast of a40r0 is southwest of a41r0.

northwest of a42r0 is southeast of a43r0.

northwest of a44r1 is southeast of a45r1.

southwest of a46r1 is northeast of a47r1.

southwest of a48r1 is northeast of a49r1.

southeast of a50r1 is northwest of a51r1.

southwest of a52r1 is northeast of a53r1.

west of a54r1 is east of a55r1.

east of a56r1 is west of a57r1.

east of a58r1 is west of a59r1.

northeast of a60r1 is southwest of a61r1.

northeast of a62r1 is southwest of a63r1.

northwest of a64r1 is southeast of a65r1.

northwest of a66r1 is southeast of a67r1.

southeast of a68r1 is northwest of a69r1.

southeast of a70r1 is northwest of a71r1.

northeast of a72r1 is southwest of a73r1.

northeast of a61r1 is southwest of a74r1.

northeast of a63r1 is southwest of a75r1.

northwest of a65r1 is southeast of a76r1.

northwest of a67r1 is southeast of a77r1.

northwest of a19r0 is southeast of a78r0.

southeast of a79r0 is northwest of a80r0.

southeast of a21r0 is northwest of a81r0.

southeast of a23r0 is northwest of a82r0.

southwest of a25r0 is northeast of a83r0.

southeast of a27r0 is northwest of a84r0.

southwest of a1r0 is northeast of a85r0.

southwest of a3r0 is northeast of a78r0.

south of a5r0 is north of a86r0.

southwest of a37r0 is northeast of a89r0.

southeast of a39r0 is northwest of a89r0.

northeast of a41r0 is southwest of a81r0.

northwest of a43r0 is southeast of a83r0.

northeast of a13r0 is southwest of a84r0.

northeast of a15r0 is southwest of a82r0.

northwest of a17r0 is southeast of a85r0.

southwest of a53r1 is northeast of a90r1.

north of a55r1 is south of a91r1.

north of a57r1 is south of a92r1.

north of a59r1 is south of a93r1.

southeast of a29r1 is northwest of a94r1.

west of a31r1 is east of a95r1.

northeast of a33r1 is southwest of a96r1.

northwest of a35r1 is southeast of a96r1.

southeast of a69r1 is northwest of a75r1.

southeast of a71r1 is northwest of a74r1.

northeast of a73r1 is southwest of a97r1.

northwest of a45r1 is southeast of a90r1.

southwest of a47r1 is northeast of a76r1.

southwest of a49r1 is northeast of a77r1.

southeast of a51r1 is northwest of a97r1.

south of a7r0 is north of a98r0.

south of a9r0 is north of a99r0.

south of a11r0 is north of a79r0.

west of a85r0 is east of a99r0.

west of a77r1 is east of a91r1.

west of a90r1 is east of a100r1.

east of a74r1 is west of a93r1.

east of a75r1 is west of a101r1.

east of a97r1 is west of a102r1.

west of a76r1 is east of a102r1.

west of a78r0 is east of a103r0.

west of a86r0 is east of a104r0.

east of a84r0 is west of a98r0.

east of a80r0 is west of a83r0.

southwest of a99r0 is northeast of a105r0.

west of a105r0 is east of a106r0.

east of a82r0 is west of a103r0.

west of a79r0 is east of a86r0.

south of a89r0 is north of a107r0.

south of a106r0 is north of a108r0.

south of a104r0 is north of a109r0.

northeast of a87r0 is southwest of a103r0.

south of a80r0 is north of a110r0.

northwest of a87r0 is southeast of a105r0.

east of a81r0 is west of a104r0.

southeast of a98r0 is northwest of a106r0.

southwest of a94r1 is northeast of a111r1.

west of a111r1 is east of a112r1.

southwest of a88r1 is northeast of a94r1.

west of a107r0 is east of a109r0.

west of a108r0 is east of a110r0.

west of a92r1 is east of a101r1.

northwest of a91r1 is southeast of a111r1.

east of a92r1 is west of a100r1.

northeast of a93r1 is southwest of a112r1.

southeast of a88r1 is northwest of a102r1.

south of a95r1 is north of a100r1.

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
the player is in a9r0.

the orientation of the player is 6.
