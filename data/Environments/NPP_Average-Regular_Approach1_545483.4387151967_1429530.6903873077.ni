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

[create a3r0]
a3r0 is a area. "An area (3) in r0".
the printed name of the a3r0 is "Room 0".
Understand "Area 3 in Room 0" as a3r0.
the parent of the a3r0 is r0.
the X of the a3r0 is "546969.3445136291".
the Y of the a3r0 is "1429862.1566995913".

[create a4r0]
a4r0 is a area. "An area (4) in r0".
the printed name of the a4r0 is "Room 0".
Understand "Area 4 in Room 0" as a4r0.
the parent of the a4r0 is r0.
the X of the a4r0 is "546989.3517621404".
the Y of the a4r0 is "1429615.2785383724".

[create a5r0]
a5r0 is a area. "An area (5) in r0".
the printed name of the a5r0 is "Room 0".
Understand "Area 5 in Room 0" as a5r0.
the parent of the a5r0 is r0.
the X of the a5r0 is "546967.1724611709".
the Y of the a5r0 is "1429943.4000000001".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "546965.6351086956".
the Y of the a6r0 is "1430016.2".

[create a7r0]
a7r0 is a area. "An area (7) in r0".
the printed name of the a7r0 is "Room 0".
Understand "Area 7 in Room 0" as a7r0.
the parent of the a7r0 is r0.
the X of the a7r0 is "547207.459199728".
the Y of the a7r0 is "1429939.0".

[create a8r0]
a8r0 is a area. "An area (8) in r0".
the printed name of the a8r0 is "Room 0".
Understand "Area 8 in Room 0" as a8r0.
the parent of the a8r0 is r0.
the X of the a8r0 is "547206.0377451611".
the Y of the a8r0 is "1430012.3333333333".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "546966.4220652174".
the Y of the a9r0 is "1430089.0".

[create a10r0]
a10r0 is a area. "An area (10) in r0".
the printed name of the a10r0 is "Room 0".
Understand "Area 10 in Room 0" as a10r0.
the parent of the a10r0 is r0.
the X of the a10r0 is "547204.6167641326".
the Y of the a10r0 is "1430085.6666666665".

[create a11r0]
a11r0 is a area. "An area (11) in r0".
the printed name of the a11r0 is "Room 0".
Understand "Area 11 in Room 0" as a11r0.
the parent of the a11r0 is r0.
the X of the a11r0 is "546967.1220652174".
the Y of the a11r0 is "1430161.7999999998".

[create a12r0]
a12r0 is a area. "An area (12) in r0".
the printed name of the a12r0 is "Room 0".
Understand "Area 12 in Room 0" as a12r0.
the parent of the a12r0 is r0.
the X of the a12r0 is "547203.19627006".
the Y of the a12r0 is "1430159.0".

[create a16r0]
a16r0 is a area. "An area (16) in r0".
the printed name of the a16r0 is "Room 0".
Understand "Area 16 in Room 0" as a16r0.
the parent of the a16r0 is r0.
the X of the a16r0 is "546891.3237452032".
the Y of the a16r0 is "1429506.3646405656".

[create a17r0]
a17r0 is a area. "An area (17) in r0".
the printed name of the a17r0 is "Room 0".
Understand "Area 17 in Room 0" as a17r0.
the parent of the a17r0 is r0.
the X of the a17r0 is "546945.7779347856".
the Y of the a17r0 is "1429602.5596938415".

[create a18r0]
a18r0 is a area. "An area (18) in r0".
the printed name of the a18r0 is "Room 0".
Understand "Area 18 in Room 0" as a18r0.
the parent of the a18r0 is r0.
the X of the a18r0 is "548211.7893884667".
the Y of the a18r0 is "1430338.5".

[create a19r0]
a19r0 is a area. "An area (19) in r0".
the printed name of the a19r0 is "Room 0".
Understand "Area 19 in Room 0" as a19r0.
the parent of the a19r0 is r0.
the X of the a19r0 is "548205.0".
the Y of the a19r0 is "1429568.577830189".

[create a20r0]
a20r0 is a area. "An area (20) in r0".
the printed name of the a20r0 is "Room 0".
Understand "Area 20 in Room 0" as a20r0.
the parent of the a20r0 is r0.
the X of the a20r0 is "546888.8657919266".
the Y of the a20r0 is "1429620.4354443646".

[create a21r0]
a21r0 is a area. "An area (21) in r0".
the printed name of the a21r0 is "Room 0".
Understand "Area 21 in Room 0" as a21r0.
the parent of the a21r0 is r0.
the X of the a21r0 is "547142.1312088763".
the Y of the a21r0 is "1429621.5045800544".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "547567.7817693059".
the Y of the a23r0 is "1429444.9748289345".

[create a24r0]
a24r0 is a area. "An area (24) in r0".
the printed name of the a24r0 is "Room 0".
Understand "Area 24 in Room 0" as a24r0.
the parent of the a24r0 is r0.
the X of the a24r0 is "548571.7817693062".
the Y of the a24r0 is "1429444.9748289343".

[create a25r0]
a25r0 is a area. "An area (25) in r0".
the printed name of the a25r0 is "Room 0".
Understand "Area 25 in Room 0" as a25r0.
the parent of the a25r0 is r0.
the X of the a25r0 is "547823.389199255".
the Y of the a25r0 is "1429440.2597765364".

[create a42r0]
a42r0 is a area. "An area (42) in r0".
the printed name of the a42r0 is "Room 0".
Understand "Area 42 in Room 0" as a42r0.
the parent of the a42r0 is r0.
the X of the a42r0 is "547199.1392057762".
the Y of the a42r0 is "1430384.65".

[create a50r0]
a50r0 is a area. "An area (50) in r0".
the printed name of the a50r0 is "Room 0".
Understand "Area 50 in Room 0" as a50r0.
the parent of the a50r0 is r0.
the X of the a50r0 is "546967.7351086957".
the Y of the a50r0 is "1430234.6".

[create a51r0]
a51r0 is a area. "An area (51) in r0".
the printed name of the a51r0 is "Room 0".
Understand "Area 51 in Room 0" as a51r0.
the parent of the a51r0 is r0.
the X of the a51r0 is "547201.7762768727".
the Y of the a51r0 is "1430232.3333333335".

[create a52r0]
a52r0 is a area. "An area (52) in r0".
the printed name of the a52r0 is "Room 0".
Understand "Area 52 in Room 0" as a52r0.
the parent of the a52r0 is r0.
the X of the a52r0 is "546968.700326087".
the Y of the a52r0 is "1430380.2".

[create a53r0]
a53r0 is a area. "An area (53) in r0".
the printed name of the a53r0 is "Room 0".
Understand "Area 53 in Room 0" as a53r0.
the parent of the a53r0 is r0.
the X of the a53r0 is "547200.2888086643".
the Y of the a53r0 is "1430307.5500000003".

[create a54r0]
a54r0 is a area. "An area (54) in r0".
the printed name of the a54r0 is "Room 0".
Understand "Area 54 in Room 0" as a54r0.
the parent of the a54r0 is r0.
the X of the a54r0 is "547207.7860830325".
the Y of the a54r0 is "1430338.5".

[create a55r0]
a55r0 is a area. "An area (55) in r0".
the printed name of the a55r0 is "Room 0".
Understand "Area 55 in Room 0" as a55r0.
the parent of the a55r0 is r0.
the X of the a55r0 is "547886.2096686172".
the Y of the a55r0 is "1430354.2122525645".

[create a76r0]
a76r0 is a area. "An area (76) in r0".
the printed name of the a76r0 is "Room 0".
Understand "Area 76 in Room 0" as a76r0.
the parent of the a76r0 is r0.
the X of the a76r0 is "546765.0".
the Y of the a76r0 is "1429480.0".

[create a77r0]
a77r0 is a area. "An area (77) in r0".
the printed name of the a77r0 is "Room 0".
Understand "Area 77 in Room 0" as a77r0.
the parent of the a77r0 is r0.
the X of the a77r0 is "546769.6801000859".
the Y of the a77r0 is "1429457.995740458".

[create a78r0]
a78r0 is a area. "An area (78) in r0".
the printed name of the a78r0 is "Room 0".
Understand "Area 78 in Room 0" as a78r0.
the parent of the a78r0 is r0.
the X of the a78r0 is "546829.9512464551".
the Y of the a78r0 is "1429393.5956763818".

[create a79r0]
a79r0 is a area. "An area (79) in r0".
the printed name of the a79r0 is "Room 0".
Understand "Area 79 in Room 0" as a79r0.
the parent of the a79r0 is r0.
the X of the a79r0 is "546800.8670190257".
the Y of the a79r0 is "1429416.1944176543".

[create a80r0]
a80r0 is a area. "An area (80) in r0".
the printed name of the a80r0 is "Room 0".
Understand "Area 80 in Room 0" as a80r0.
the parent of the a80r0 is r0.
the X of the a80r0 is "546846.0".
the Y of the a80r0 is "1429372.0".

[create a81r0]
a81r0 is a area. "An area (81) in r0".
the printed name of the a81r0 is "Room 0".
Understand "Area 81 in Room 0" as a81r0.
the parent of the a81r0 is r0.
the X of the a81r0 is "546872.9601572268".
the Y of the a81r0 is "1429408.7964757935".

[create a82r0]
a82r0 is a area. "An area (82) in r0".
the printed name of the a82r0 is "Room 0".
Understand "Area 82 in Room 0" as a82r0.
the parent of the a82r0 is r0.
the X of the a82r0 is "546765.645455358".
the Y of the a82r0 is "1429480.1274465993".

[create a83r0]
a83r0 is a area. "An area (83) in r0".
the printed name of the a83r0 is "Room 0".
Understand "Area 83 in Room 0" as a83r0.
the parent of the a83r0 is r0.
the X of the a83r0 is "546756.9611690349".
the Y of the a83r0 is "1429522.4109719293".

[create a84r0]
a84r0 is a area. "An area (84) in r0".
the printed name of the a84r0 is "Room 0".
Understand "Area 84 in Room 0" as a84r0.
the parent of the a84r0 is r0.
the X of the a84r0 is "547926.0".
the Y of the a84r0 is "1430260.0".

[create a85r0]
a85r0 is a area. "An area (85) in r0".
the printed name of the a85r0 is "Room 0".
Understand "Area 85 in Room 0" as a85r0.
the parent of the a85r0 is r0.
the X of the a85r0 is "547889.2".
the Y of the a85r0 is "1430346.0802742287".

[create a92r0]
a92r0 is a area. "An area (92) in r0".
the printed name of the a92r0 is "Room 0".
Understand "Area 92 in Room 0" as a92r0.
the parent of the a92r0 is r0.
the X of the a92r0 is "547778.0".
the Y of the a92r0 is "1429391.0".

[create a93r0]
a93r0 is a area. "An area (93) in r0".
the printed name of the a93r0 is "Room 0".
Understand "Area 93 in Room 0" as a93r0.
the parent of the a93r0 is r0.
the X of the a93r0 is "547809.2137161085".
the Y of the a93r0 is "1429416.6339712923".

[create a96r0]
a96r0 is a area. "An area (96) in r0".
the printed name of the a96r0 is "Room 0".
Understand "Area 96 in Room 0" as a96r0.
the parent of the a96r0 is r0.
the X of the a96r0 is "547769.0".
the Y of the a96r0 is "1429480.0".

[create a97r0]
a97r0 is a area. "An area (97) in r0".
the printed name of the a97r0 is "Room 0".
Understand "Area 97 in Room 0" as a97r0.
the parent of the a97r0 is r0.
the X of the a97r0 is "547800.0059757693".
the Y of the a97r0 is "1429460.6579076622".

[create a100r0]
a100r0 is a area. "An area (100) in r0".
the printed name of the a100r0 is "Room 0".
Understand "Area 100 in Room 0" as a100r0.
the parent of the a100r0 is r0.
the X of the a100r0 is "547616.0".
the Y of the a100r0 is "1429391.0".

[create a101r0]
a101r0 is a area. "An area (101) in r0".
the printed name of the a101r0 is "Room 0".
Understand "Area 101 in Room 0" as a101r0.
the parent of the a101r0 is r0.
the X of the a101r0 is "547584.7862838916".
the Y of the a101r0 is "1429416.633971292".

[create a102r0]
a102r0 is a area. "An area (102) in r0".
the printed name of the a102r0 is "Room 0".
Understand "Area 102 in Room 0" as a102r0.
the parent of the a102r0 is r0.
the X of the a102r0 is "548620.0".
the Y of the a102r0 is "1429391.0".

[create a103r0]
a103r0 is a area. "An area (103) in r0".
the printed name of the a103r0 is "Room 0".
Understand "Area 103 in Room 0" as a103r0.
the parent of the a103r0 is r0.
the X of the a103r0 is "548588.7862838916".
the Y of the a103r0 is "1429416.633971292".

[create a104r0]
a104r0 is a area. "An area (104) in r0".
the printed name of the a104r0 is "Room 0".
Understand "Area 104 in Room 0" as a104r0.
the parent of the a104r0 is r0.
the X of the a104r0 is "547625.0".
the Y of the a104r0 is "1429480.0".

[create a105r0]
a105r0 is a area. "An area (105) in r0".
the printed name of the a105r0 is "Room 0".
Understand "Area 105 in Room 0" as a105r0.
the parent of the a105r0 is r0.
the X of the a105r0 is "547582.1485252038".
the Y of the a105r0 is "1429461.8557671143".

[create a106r0]
a106r0 is a area. "An area (106) in r0".
the printed name of the a106r0 is "Room 0".
Understand "Area 106 in Room 0" as a106r0.
the parent of the a106r0 is r0.
the X of the a106r0 is "548629.0".
the Y of the a106r0 is "1429480.0".

[create a107r0]
a107r0 is a area. "An area (107) in r0".
the printed name of the a107r0 is "Room 0".
Understand "Area 107 in Room 0" as a107r0.
the parent of the a107r0 is r0.
the X of the a107r0 is "548586.1485252036".
the Y of the a107r0 is "1429461.8557671143".

[create a113r0]
a113r0 is a area. "An area (113) in r0".
the printed name of the a113r0 is "Room 0".
Understand "Area 113 in Room 0" as a113r0.
the parent of the a113r0 is r0.
the X of the a113r0 is "546833.674865904".
the Y of the a113r0 is "1429463.523071742".

[create a0r1]
a0r1 is a area. "An area (0) in r1".
the printed name of the a0r1 is "Room 1".
Understand "Area 0 in Room 1" as a0r1.
the parent of the a0r1 is r1.
the X of the a0r1 is "547153.7453558876".
the Y of the a0r1 is "1431287.6756333306".

[create a1r1]
a1r1 is a area. "An area (1) in r1".
the printed name of the a1r1 is "Room 1".
Understand "Area 1 in Room 1" as a1r1.
the parent of the a1r1 is r1.
the X of the a1r1 is "547289.5726181442".
the Y of the a1r1 is "1431402.9819553911".

[create a2r1]
a2r1 is a area. "An area (2) in r1".
the printed name of the a2r1 is "Room 1".
Understand "Area 2 in Room 1" as a2r1.
the parent of the a2r1 is r1.
the X of the a2r1 is "547838.175928817".
the Y of the a2r1 is "1431443.5425642629".

[create a26r1]
a26r1 is a area. "An area (26) in r1".
the printed name of the a26r1 is "Room 1".
Understand "Area 26 in Room 1" as a26r1.
the parent of the a26r1 is r1.
the X of the a26r1 is "548185.4507751702".
the Y of the a26r1 is "1431312.6114864924".

[create a27r1]
a27r1 is a area. "An area (27) in r1".
the printed name of the a27r1 is "Room 1".
Understand "Area 27 in Room 1" as a27r1.
the parent of the a27r1 is r1.
the X of the a27r1 is "548214.7674276347".
the Y of the a27r1 is "1431128.751907786".

[create a29r1]
a29r1 is a area. "An area (29) in r1".
the printed name of the a29r1 is "Room 1".
Understand "Area 29 in Room 1" as a29r1.
the parent of the a29r1 is r1.
the X of the a29r1 is "548972.5663719163".
the Y of the a29r1 is "1431391.0382126388".

[create a30r1]
a30r1 is a area. "An area (30) in r1".
the printed name of the a30r1 is "Room 1".
Understand "Area 30 in Room 1" as a30r1.
the parent of the a30r1 is r1.
the X of the a30r1 is "546612.4446009535".
the Y of the a30r1 is "1431464.1811858914".

[create a31r1]
a31r1 is a area. "An area (31) in r1".
the printed name of the a31r1 is "Room 1".
Understand "Area 31 in Room 1" as a31r1.
the parent of the a31r1 is r1.
the X of the a31r1 is "546834.4738955821".
the Y of the a31r1 is "1431440.62248996".

[create a32r1]
a32r1 is a area. "An area (32) in r1".
the printed name of the a32r1 is "Room 1".
Understand "Area 32 in Room 1" as a32r1.
the parent of the a32r1 is r1.
the X of the a32r1 is "547051.5873019714".
the Y of the a32r1 is "1431298.650409535".

[create a34r1]
a34r1 is a area. "An area (34) in r1".
the printed name of the a34r1 is "Room 1".
Understand "Area 34 in Room 1" as a34r1.
the parent of the a34r1 is r1.
the X of the a34r1 is "547198.9926353791".
the Y of the a34r1 is "1430693.05".

[create a35r1]
a35r1 is a area. "An area (35) in r1".
the printed name of the a35r1 is "Room 1".
Understand "Area 35 in Room 1" as a35r1.
the parent of the a35r1 is r1.
the X of the a35r1 is "547199.2596028881".
the Y of the a35r1 is "1430770.15".

[create a36r1]
a36r1 is a area. "An area (36) in r1".
the printed name of the a36r1 is "Room 1".
Understand "Area 36 in Room 1" as a36r1.
the parent of the a36r1 is r1.
the X of the a36r1 is "549223.7894678493".
the Y of the a36r1 is "1431112.9604600752".

[create a37r1]
a37r1 is a area. "An area (37) in r1".
the printed name of the a37r1 is "Room 1".
Understand "Area 37 in Room 1" as a37r1.
the parent of the a37r1 is r1.
the X of the a37r1 is "549224.3229297196".
the Y of the a37r1 is "1430741.76926021".

[create a38r1]
a38r1 is a area. "An area (38) in r1".
the printed name of the a38r1 is "Room 1".
Understand "Area 38 in Room 1" as a38r1.
the parent of the a38r1 is r1.
the X of the a38r1 is "547199.6480144404".
the Y of the a38r1 is "1430847.2500000002".

[create a39r1]
a39r1 is a area. "An area (39) in r1".
the printed name of the a39r1 is "Room 1".
Understand "Area 39 in Room 1" as a39r1.
the parent of the a39r1 is r1.
the X of the a39r1 is "548216.5676380108".
the Y of the a39r1 is "1430738.7532928498".

[create a40r1]
a40r1 is a area. "An area (40) in r1".
the printed name of the a40r1 is "Room 1".
Understand "Area 40 in Room 1" as a40r1.
the parent of the a40r1 is r1.
the X of the a40r1 is "547200.1578700361".
the Y of the a40r1 is "1430924.3499999999".

[create a41r1]
a41r1 is a area. "An area (41) in r1".
the printed name of the a41r1 is "Room 1".
Understand "Area 41 in Room 1" as a41r1.
the parent of the a41r1 is r1.
the X of the a41r1 is "547200.7891696751".
the Y of the a41r1 is "1431001.45".

[create a43r1]
a43r1 is a area. "An area (43) in r1".
the printed name of the a43r1 is "Room 1".
Understand "Area 43 in Room 1" as a43r1.
the parent of the a43r1 is r1.
the X of the a43r1 is "547198.920397112".
the Y of the a43r1 is "1430461.75".

[create a44r1]
a44r1 is a area. "An area (44) in r1".
the printed name of the a44r1 is "Room 1".
Understand "Area 44 in Room 1" as a44r1.
the parent of the a44r1 is r1.
the X of the a44r1 is "546969.4443947934".
the Y of the a44r1 is "1430525.7999999998".

[create a45r1]
a45r1 is a area. "An area (45) in r1".
the printed name of the a45r1 is "Room 1".
Understand "Area 45 in Room 1" as a45r1.
the parent of the a45r1 is r1.
the X of the a45r1 is "546969.1490123457".
the Y of the a45r1 is "1430453.0".

[create a46r1]
a46r1 is a area. "An area (46) in r1".
the printed name of the a46r1 is "Room 1".
Understand "Area 46 in Room 1" as a46r1.
the parent of the a46r1 is r1.
the X of the a46r1 is "547198.823032491".
the Y of the a46r1 is "1430538.8499999999".

[create a47r1]
a47r1 is a area. "An area (47) in r1".
the printed name of the a47r1 is "Room 1".
Understand "Area 47 in Room 1" as a47r1.
the parent of the a47r1 is r1.
the X of the a47r1 is "546969.58647343".
the Y of the a47r1 is "1430598.6".

[create a48r1]
a48r1 is a area. "An area (48) in r1".
the printed name of the a48r1 is "Room 1".
Understand "Area 48 in Room 1" as a48r1.
the parent of the a48r1 is r1.
the X of the a48r1 is "548179.3159883294".
the Y of the a48r1 is "1430514.0912121371".

[create a49r1]
a49r1 is a area. "An area (49) in r1".
the printed name of the a49r1 is "Room 1".
Understand "Area 49 in Room 1" as a49r1.
the parent of the a49r1 is r1.
the X of the a49r1 is "547198.8471119135".
the Y of the a49r1 is "1430615.95".

[create a56r1]
a56r1 is a area. "An area (56) in r1".
the printed name of the a56r1 is "Room 1".
Understand "Area 56 in Room 1" as a56r1.
the parent of the a56r1 is r1.
the X of the a56r1 is "546796.0".
the Y of the a56r1 is "1431514.0".

[create a57r1]
a57r1 is a area. "An area (57) in r1".
the printed name of the a57r1 is "Room 1".
Understand "Area 57 in Room 1" as a57r1.
the parent of the a57r1 is r1.
the X of the a57r1 is "546827.8264604809".
the Y of the a57r1 is "1431505.767353952".

[create a58r1]
a58r1 is a area. "An area (58) in r1".
the printed name of the a58r1 is "Room 1".
Understand "Area 58 in Room 1" as a58r1.
the parent of the a58r1 is r1.
the X of the a58r1 is "547333.0".
the Y of the a58r1 is "1431504.0".

[create a59r1]
a59r1 is a area. "An area (59) in r1".
the printed name of the a59r1 is "Room 1".
Understand "Area 59 in Room 1" as a59r1.
the parent of the a59r1 is r1.
the X of the a59r1 is "547300.0059829059".
the Y of the a59r1 is "1431505.7505982905".

[create a60r1]
a60r1 is a area. "An area (60) in r1".
the printed name of the a60r1 is "Room 1".
Understand "Area 60 in Room 1" as a60r1.
the parent of the a60r1 is r1.
the X of the a60r1 is "548358.5361148497".
the Y of the a60r1 is "1431188.525945865".

[create a61r1]
a61r1 is a area. "An area (61) in r1".
the printed name of the a61r1 is "Room 1".
Understand "Area 61 in Room 1" as a61r1.
the parent of the a61r1 is r1.
the X of the a61r1 is "548326.8539864278".
the Y of the a61r1 is "1431143.2142857143".

[create a62r1]
a62r1 is a area. "An area (62) in r1".
the printed name of the a62r1 is "Room 1".
Understand "Area 62 in Room 1" as a62r1.
the parent of the a62r1 is r1.
the X of the a62r1 is "549016.0".
the Y of the a62r1 is "1431504.0".

[create a63r1]
a63r1 is a area. "An area (63) in r1".
the printed name of the a63r1 is "Room 1".
Understand "Area 63 in Room 1" as a63r1.
the parent of the a63r1 is r1.
the X of the a63r1 is "548982.2".
the Y of the a63r1 is "1431415.6469450549".

[create a68r1]
a68r1 is a area. "An area (68) in r1".
the printed name of the a68r1 is "Room 1".
Understand "Area 68 in Room 1" as a68r1.
the parent of the a68r1 is r1.
the X of the a68r1 is "546558.0".
the Y of the a68r1 is "1431424.0".

[create a69r1]
a69r1 is a area. "An area (69) in r1".
the printed name of the a69r1 is "Room 1".
Understand "Area 69 in Room 1" as a69r1.
the parent of the a69r1 is r1.
the X of the a69r1 is "546585.2059471365".
the Y of the a69r1 is "1431444.0044052864".

[create a72r1]
a72r1 is a area. "An area (72) in r1".
the printed name of the a72r1 is "Room 1".
Understand "Area 72 in Room 1" as a72r1.
the parent of the a72r1 is r1.
the X of the a72r1 is "546567.0".
the Y of the a72r1 is "1431514.0".

[create a73r1]
a73r1 is a area. "An area (73) in r1".
the printed name of the a73r1 is "Room 1".
Understand "Area 73 in Room 1" as a73r1.
the parent of the a73r1 is r1.
the X of the a73r1 is "546597.9153780069".
the Y of the a73r1 is "1431488.1834621995".

[create a74r1]
a74r1 is a area. "An area (74) in r1".
the printed name of the a74r1 is "Room 1".
Understand "Area 74 in Room 1" as a74r1.
the parent of the a74r1 is r1.
the X of the a74r1 is "547799.0".
the Y of the a74r1 is "1431504.0".

[create a75r1]
a75r1 is a area. "An area (75) in r1".
the printed name of the a75r1 is "Room 1".
Understand "Area 75 in Room 1" as a75r1.
the parent of the a75r1 is r1.
the X of the a75r1 is "547831.826460481".
the Y of the a75r1 is "1431505.7673539517".

[create a86r1]
a86r1 is a area. "An area (86) in r1".
the printed name of the a86r1 is "Room 1".
Understand "Area 86 in Room 1" as a86r1.
the parent of the a86r1 is r1.
the X of the a86r1 is "549373.0".
the Y of the a86r1 is "1430678.0".

[create a87r1]
a87r1 is a area. "An area (87) in r1".
the printed name of the a87r1 is "Room 1".
Understand "Area 87 in Room 1" as a87r1.
the parent of the a87r1 is r1.
the X of the a87r1 is "549286.8956183491".
the Y of the a87r1 is "1430713.7857142857".

[create a88r1]
a88r1 is a area. "An area (88) in r1".
the printed name of the a88r1 is "Room 1".
Understand "Area 88 in Room 1" as a88r1.
the parent of the a88r1 is r1.
the X of the a88r1 is "548369.0".
the Y of the a88r1 is "1430678.0".

[create a89r1]
a89r1 is a area. "An area (89) in r1".
the printed name of the a89r1 is "Room 1".
Understand "Area 89 in Room 1" as a89r1.
the parent of the a89r1 is r1.
the X of the a89r1 is "548311.6048216443".
the Y of the a89r1 is "1430713.7857142857".

[create a90r1]
a90r1 is a area. "An area (90) in r1".
the printed name of the a90r1 is "Room 1".
Understand "Area 90 in Room 1" as a90r1.
the parent of the a90r1 is r1.
the X of the a90r1 is "549373.0".
the Y of the a90r1 is "1431179.0".

[create a91r1]
a91r1 is a area. "An area (91) in r1".
the printed name of the a91r1 is "Room 1".
Understand "Area 91 in Room 1" as a91r1.
the parent of the a91r1 is r1.
the X of the a91r1 is "549293.788515406".
the Y of the a91r1 is "1431143.2142857143".

[create a112r1]
a112r1 is a area. "An area (112) in r1".
the printed name of the a112r1 is "Room 1".
Understand "Area 112 in Room 1" as a112r1.
the parent of the a112r1 is r1.
the X of the a112r1 is "546972.9480000003".
the Y of the a112r1 is "1430598.5999999999".

[create a15r2]
a15r2 is a area. "An area (15) in r2".
the printed name of the a15r2 is "Room 2".
Understand "Area 15 in Room 2" as a15r2.
the parent of the a15r2 is r2.
the X of the a15r2 is "545554.036572359".
the Y of the a15r2 is "1431232.6014533546".

[create a28r2]
a28r2 is a area. "An area (28) in r2".
the printed name of the a28r2 is "Room 2".
Understand "Area 28 in Room 2" as a28r2.
the parent of the a28r2 is r2.
the X of the a28r2 is "545503.89814202".
the Y of the a28r2 is "1431356.089074554".

[create a33r2]
a33r2 is a area. "An area (33) in r2".
the printed name of the a33r2 is "Room 2".
Understand "Area 33 in Room 2" as a33r2.
the parent of the a33r2 is r2.
the X of the a33r2 is "545168.0926719012".
the Y of the a33r2 is "1431459.8065726864".

[create a64r2]
a64r2 is a area. "An area (64) in r2".
the printed name of the a64r2 is "Room 2".
Understand "Area 64 in Room 2" as a64r2.
the parent of the a64r2 is r2.
the X of the a64r2 is "545578.0".
the Y of the a64r2 is "1431504.0".

[create a65r2]
a65r2 is a area. "An area (65) in r2".
the printed name of the a65r2 is "Room 2".
Understand "Area 65 in Room 2" as a65r2.
the parent of the a65r2 is r2.
the X of the a65r2 is "545544.875".
the Y of the a65r2 is "1431430.0006598176".

[create a66r2]
a66r2 is a area. "An area (66) in r2".
the printed name of the a66r2 is "Room 2".
Understand "Area 66 in Room 2" as a66r2.
the parent of the a66r2 is r2.
the X of the a66r2 is "545111.0".
the Y of the a66r2 is "1431424.0".

[create a67r2]
a67r2 is a area. "An area (67) in r2".
the printed name of the a67r2 is "Room 2".
Understand "Area 67 in Room 2" as a67r2.
the parent of the a67r2 is r2.
the X of the a67r2 is "545153.06".
the Y of the a67r2 is "1431442.519".

[create a70r2]
a70r2 is a area. "An area (70) in r2".
the printed name of the a70r2 is "Room 2".
Understand "Area 70 in Room 2" as a70r2.
the parent of the a70r2 is r2.
the X of the a70r2 is "545120.0".
the Y of the a70r2 is "1431514.0".

[create a71r2]
a71r2 is a area. "An area (71) in r2".
the printed name of the a71r2 is "Room 2".
Understand "Area 71 in Room 2" as a71r2.
the parent of the a71r2 is r2.
the X of the a71r2 is "545150.9153780069".
the Y of the a71r2 is "1431488.1834621993".

[create a13r3]
a13r3 is a area. "An area (13) in r3".
the printed name of the a13r3 is "Room 3".
Understand "Area 13 in Room 3" as a13r3.
the parent of the a13r3 is r3.
the X of the a13r3 is "545119.3147627444".
the Y of the a13r3 is "1429440.2898139763".

[create a14r3]
a14r3 is a area. "An area (14) in r3".
the printed name of the a14r3 is "Room 3".
Understand "Area 14 in Room 3" as a14r3.
the parent of the a14r3 is r3.
the X of the a14r3 is "545483.4387151967".
the Y of the a14r3 is "1429530.6903873077".

[create a22r3]
a22r3 is a area. "An area (22) in r3".
the printed name of the a22r3 is "Room 3".
Understand "Area 22 in Room 3" as a22r3.
the parent of the a22r3 is r3.
the X of the a22r3 is "545839.7586021505".
the Y of the a22r3 is "1429440.0134408604".

[create a94r3]
a94r3 is a area. "An area (94) in r3".
the printed name of the a94r3 is "Room 3".
Understand "Area 94 in Room 3" as a94r3.
the parent of the a94r3 is r3.
the X of the a94r3 is "545894.0".
the Y of the a94r3 is "1429480.0".

[create a95r3]
a95r3 is a area. "An area (95) in r3".
the printed name of the a95r3 is "Room 3".
Understand "Area 95 in Room 3" as a95r3.
the parent of the a95r3 is r3.
the X of the a95r3 is "545866.849158728".
the Y of the a95r3 is "1429460.2680626004".

[create a98r3]
a98r3 is a area. "An area (98) in r3".
the printed name of the a98r3 is "Room 3".
Understand "Area 98 in Room 3" as a98r3.
the parent of the a98r3 is r3.
the X of the a98r3 is "545065.0".
the Y of the a98r3 is "1429480.0".

[create a99r3]
a99r3 is a area. "An area (99) in r3".
the printed name of the a99r3 is "Room 3".
Understand "Area 99 in Room 3" as a99r3.
the parent of the a99r3 is r3.
the X of the a99r3 is "545096.6769635527".
the Y of the a99r3 is "1429460.7257603593".

[create a108r3]
a108r3 is a area. "An area (108) in r3".
the printed name of the a108r3 is "Room 3".
Understand "Area 108 in Room 3" as a108r3.
the parent of the a108r3 is r3.
the X of the a108r3 is "545885.0".
the Y of the a108r3 is "1429391.0".

[create a109r3]
a109r3 is a area. "An area (109) in r3".
the printed name of the a109r3 is "Room 3".
Understand "Area 109 in Room 3" as a109r3.
the parent of the a109r3 is r3.
the X of the a109r3 is "545853.7862838916".
the Y of the a109r3 is "1429416.633971292".

[create a110r3]
a110r3 is a area. "An area (110) in r3".
the printed name of the a110r3 is "Room 3".
Understand "Area 110 in Room 3" as a110r3.
the parent of the a110r3 is r3.
the X of the a110r3 is "545074.0".
the Y of the a110r3 is "1429391.0".

[create a111r3]
a111r3 is a area. "An area (111) in r3".
the printed name of the a111r3 is "Room 3".
Understand "Area 111 in Room 3" as a111r3.
the parent of the a111r3 is r3.
the X of the a111r3 is "545105.0420577895".
the Y of the a111r3 is "1429416.6166125853".

[create d0]
d0 is a door. "A door between a15r2 and a14r3".
d0 is south of a15r2 and north of a14r3.

[create d2]
d2 is a door. "A door between a20r0 and a15r2".
d2 is northwest of a20r0 and southeast of a15r2.

[create d4]
d4 is a door. "A door between a42r0 and a43r1".
d4 is north of a42r0 and south of a43r1.

[create d5]
d5 is a door. "A door between a55r0 and a48r1".
d5 is northeast of a55r0 and southwest of a48r1.

[create d6]
d6 is a door. "A door between a52r0 and a45r1".
d6 is north of a52r0 and south of a45r1.

[create d7]
d7 is a door. "A door between a18r0 and a48r1".
d7 is north of a18r0 and south of a48r1.

[create d8]
d8 is a door. "A door between a18r0 and a37r1".
d8 is east of a18r0 and west of a37r1.

northeast of a0r1 is southwest of a1r1.

east of a1r1 is west of a2r1.

south of a3r0 is north of a4r0.

north of a3r0 is south of a5r0.

north of a5r0 is south of a6r0.

north of a7r0 is south of a8r0.

north of a6r0 is south of a9r0.

north of a8r0 is south of a10r0.

north of a9r0 is south of a11r0.

north of a10r0 is south of a12r0.

east of a13r3 is west of a14r3.

northeast of a16r0 is southwest of a17r0.

south of a18r0 is north of a19r0.

west of a4r0 is east of a17r0.

east of a4r0 is west of a21r0.

south of a7r0 is north of a21r0.

east of a14r3 is west of a22r3.

southeast of a21r0 is northwest of a23r0.

east of a19r0 is west of a24r0.

west of a19r0 is east of a25r0.

south of a26r1 is north of a27r1.

north of a15r2 is south of a28r2.

east of a2r1 is west of a26r1.

east of a26r1 is west of a29r1.

east of a30r1 is west of a31r1.

west of a0r1 is east of a32r1.

southeast of a31r1 is northwest of a32r1.

west of a28r2 is east of a33r2.

north of a34r1 is south of a35r1.

south of a36r1 is north of a37r1.

north of a35r1 is south of a38r1.

south of a27r1 is north of a39r1.

north of a38r1 is south of a40r1.

north of a40r1 is south of a41r1.

south of a0r1 is north of a41r1.

southeast of a29r1 is northwest of a36r1.

south of a44r1 is north of a45r1.

north of a43r1 is south of a46r1.

north of a44r1 is south of a47r1.

south of a39r1 is north of a48r1.

north of a46r1 is south of a49r1.

south of a32r1 is north of a47r1.

south of a34r1 is north of a49r1.

north of a11r0 is south of a50r0.

north of a12r0 is south of a51r0.

north of a50r0 is south of a52r0.

north of a51r0 is south of a53r0.

east of a54r0 is west of a55r0.

east of a56r1 is west of a57r1.

west of a58r1 is east of a59r1.

southwest of a60r1 is northeast of a61r1.

south of a62r1 is north of a63r1.

southwest of a64r2 is northeast of a65r2.

northeast of a66r2 is southwest of a67r2.

northeast of a68r1 is southwest of a69r1.

southeast of a70r2 is northwest of a71r2.

southeast of a72r1 is northwest of a73r1.

east of a74r1 is west of a75r1.

south of a76r0 is north of a77r0.

northwest of a78r0 is southeast of a79r0.

northeast of a80r0 is southwest of a81r0.

north of a82r0 is south of a83r0.

northwest of a84r0 is southeast of a85r0.

northwest of a86r1 is southeast of a87r1.

northwest of a88r1 is southeast of a89r1.

southwest of a90r1 is northeast of a91r1.

northeast of a92r0 is southwest of a93r0.

southwest of a94r3 is northeast of a95r3.

southeast of a96r0 is northwest of a97r0.

southeast of a98r3 is northwest of a99r3.

northwest of a100r0 is southeast of a101r0.

northwest of a102r0 is southeast of a103r0.

southwest of a104r0 is northeast of a105r0.

southwest of a106r0 is northeast of a107r0.

northwest of a108r3 is southeast of a109r3.

northeast of a110r3 is southwest of a111r3.

northeast of a28r2 is southwest of a65r2.

southwest of a33r2 is northeast of a67r2.

southwest of a30r1 is northeast of a69r1.

northwest of a33r2 is southeast of a71r2.

northwest of a30r1 is southeast of a73r1.

north of a2r1 is south of a75r1.

north of a31r1 is south of a57r1.

north of a1r1 is south of a59r1.

north of a53r0 is south of a54r0.

south of a42r0 is north of a54r0.

east of a47r1 is west of a112r1.

southeast of a37r1 is northwest of a87r1.

east of a39r1 is west of a89r1.

northeast of a36r1 is southwest of a91r1.

east of a27r1 is west of a61r1.

north of a29r1 is south of a63r1.

northeast of a24r0 is southwest of a107r0.

east of a77r0 is west of a113r0.

northeast of a79r0 is southwest of a113r0.

southwest of a16r0 is northeast of a113r0.

south of a16r0 is north of a81r0.

southwest of a20r0 is northeast of a83r0.

west of a17r0 is east of a20r0.

south of a55r0 is north of a85r0.

southwest of a13r3 is northeast of a111r3.

southwest of a25r0 is northeast of a93r0.

northeast of a22r3 is southwest of a95r3.

northwest of a25r0 is southeast of a97r0.

northwest of a13r3 is southeast of a99r3.

southeast of a23r0 is northwest of a101r0.

southeast of a24r0 is northwest of a103r0.

northeast of a23r0 is southwest of a105r0.

southeast of a22r3 is northwest of a109r3.

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
the player is in a14r3.

the orientation of the player is 6.
