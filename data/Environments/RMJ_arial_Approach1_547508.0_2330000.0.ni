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
the X of the a0r0 is "546701.7555245539".
the Y of the a0r0 is "2332045.892344015".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "546839.030133659".
the Y of the a1r0 is "2331976.423458389".

[create a2r0]
a2r0 is a area. "An area (2) in r0".
the printed name of the a2r0 is "Room 0".
Understand "Area 2 in Room 0" as a2r0.
the parent of the a2r0 is r0.
the X of the a2r0 is "546627.5000000001".
the Y of the a2r0 is "2332224.575163399".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "545279.5000000003".
the Y of the a9r0 is "2332201.379310345".

[create a10r0]
a10r0 is a area. "An area (10) in r0".
the printed name of the a10r0 is "Room 0".
Understand "Area 10 in Room 0" as a10r0.
the parent of the a10r0 is r0.
the X of the a10r0 is "545403.016849386".
the Y of the a10r0 is "2332079.237795951".

[create a16r0]
a16r0 is a area. "An area (16) in r0".
the printed name of the a16r0 is "Room 0".
Understand "Area 16 in Room 0" as a16r0.
the parent of the a16r0 is r0.
the X of the a16r0 is "546695.1392057586".
the Y of the a16r0 is "2331221.363580644".

[create a17r0]
a17r0 is a area. "An area (17) in r0".
the printed name of the a17r0 is "Room 0".
Understand "Area 17 in Room 0" as a17r0.
the parent of the a17r0 is r0.
the X of the a17r0 is "546795.443688029".
the Y of the a17r0 is "2331493.207870887".

[create a18r0]
a18r0 is a area. "An area (18) in r0".
the printed name of the a18r0 is "Room 0".
Understand "Area 18 in Room 0" as a18r0.
the parent of the a18r0 is r0.
the X of the a18r0 is "546999.0537436341".
the Y of the a18r0 is "2331440.2484276732".

[create a19r0]
a19r0 is a area. "An area (19) in r0".
the printed name of the a19r0 is "Room 0".
Understand "Area 19 in Room 0" as a19r0.
the parent of the a19r0 is r0.
the X of the a19r0 is "547128.3173417337".
the Y of the a19r0 is "2331530.252352863".

[create a20r0]
a20r0 is a area. "An area (20) in r0".
the printed name of the a20r0 is "Room 0".
Understand "Area 20 in Room 0" as a20r0.
the parent of the a20r0 is r0.
the X of the a20r0 is "547152.3985383252".
the Y of the a20r0 is "2331457.454416266".

[create a21r0]
a21r0 is a area. "An area (21) in r0".
the printed name of the a21r0 is "Room 0".
Understand "Area 21 in Room 0" as a21r0.
the parent of the a21r0 is r0.
the X of the a21r0 is "546876.3785840498".
the Y of the a21r0 is "2331786.2702633445".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "546992.5000000002".
the Y of the a22r0 is "2332188.0236277874".

[create a28r0]
a28r0 is a area. "An area (28) in r0".
the printed name of the a28r0 is "Room 0".
Understand "Area 28 in Room 0" as a28r0.
the parent of the a28r0 is r0.
the X of the a28r0 is "545415.5758874029".
the Y of the a28r0 is "2331152.0".

[create a47r0]
a47r0 is a area. "An area (47) in r0".
the printed name of the a47r0 is "Room 0".
Understand "Area 47 in Room 0" as a47r0.
the parent of the a47r0 is r0.
the X of the a47r0 is "547000.0".
the Y of the a47r0 is "2331411.0".

[create a48r0]
a48r0 is a area. "An area (48) in r0".
the printed name of the a48r0 is "Room 0".
Understand "Area 48 in Room 0" as a48r0.
the parent of the a48r0 is r0.
the X of the a48r0 is "547037.4627199855".
the Y of the a48r0 is "2331419.242612574".

[create a49r0]
a49r0 is a area. "An area (49) in r0".
the printed name of the a49r0 is "Room 0".
Understand "Area 49 in Room 0" as a49r0.
the parent of the a49r0 is r0.
the X of the a49r0 is "547000.0".
the Y of the a49r0 is "2331527.993710692".

[create a50r0]
a50r0 is a area. "An area (50) in r0".
the printed name of the a50r0 is "Room 0".
Understand "Area 50 in Room 0" as a50r0.
the parent of the a50r0 is r0.
the X of the a50r0 is "547033.548435113".
the Y of the a50r0 is "2331498.745283019".

[create a53r0]
a53r0 is a area. "An area (53) in r0".
the printed name of the a53r0 is "Room 0".
Understand "Area 53 in Room 0" as a53r0.
the parent of the a53r0 is r0.
the X of the a53r0 is "547029.0".
the Y of the a53r0 is "2332240.0".

[create a54r0]
a54r0 is a area. "An area (54) in r0".
the printed name of the a54r0 is "Room 0".
Understand "Area 54 in Room 0" as a54r0.
the parent of the a54r0 is r0.
the X of the a54r0 is "546591.0".
the Y of the a54r0 is "2332240.0".

[create a58r0]
a58r0 is a area. "An area (58) in r0".
the printed name of the a58r0 is "Room 0".
Understand "Area 58 in Room 0" as a58r0.
the parent of the a58r0 is r0.
the X of the a58r0 is "545242.0".
the Y of the a58r0 is "2332240.0".

[create a32r1]
a32r1 is a area. "An area (32) in r1".
the printed name of the a32r1 is "Room 1".
Understand "Area 32 in Room 1" as a32r1.
the parent of the a32r1 is r1.
the X of the a32r1 is "547660.1926899109".
the Y of the a32r1 is "2330157.834122907".

[create a34r1]
a34r1 is a area. "An area (34) in r1".
the printed name of the a34r1 is "Room 1".
Understand "Area 34 in Room 1" as a34r1.
the parent of the a34r1 is r1.
the X of the a34r1 is "547619.971707561".
the Y of the a34r1 is "2330159.864204035".

[create a39r1]
a39r1 is a area. "An area (39) in r1".
the printed name of the a39r1 is "Room 1".
Understand "Area 39 in Room 1" as a39r1.
the parent of the a39r1 is r1.
the X of the a39r1 is "547540.75".
the Y of the a39r1 is "2330051.2901447318".

[create a40r1]
a40r1 is a area. "An area (40) in r1".
the printed name of the a40r1 is "Room 1".
Understand "Area 40 in Room 1" as a40r1.
the parent of the a40r1 is r1.
the X of the a40r1 is "547737.25".
the Y of the a40r1 is "2330051.484325186".

[create a65r1]
a65r1 is a area. "An area (65) in r1".
the printed name of the a65r1 is "Room 1".
Understand "Area 65 in Room 1" as a65r1.
the parent of the a65r1 is r1.
the X of the a65r1 is "547508.0".
the Y of the a65r1 is "2330000.0".

[create a66r1]
a66r1 is a area. "An area (66) in r1".
the printed name of the a66r1 is "Room 1".
Understand "Area 66 in Room 1" as a66r1.
the parent of the a66r1 is r1.
the X of the a66r1 is "547770.0".
the Y of the a66r1 is "2330000.0".

[create a5r2]
a5r2 is a area. "An area (5) in r2".
the printed name of the a5r2 is "Room 2".
Understand "Area 5 in Room 2" as a5r2.
the parent of the a5r2 is r2.
the X of the a5r2 is "548426.0511527918".
the Y of the a5r2 is "2332061.748260341".

[create a6r2]
a6r2 is a area. "An area (6) in r2".
the printed name of the a6r2 is "Room 2".
Understand "Area 6 in Room 2" as a6r2.
the parent of the a6r2 is r2.
the X of the a6r2 is "548500.9856578638".
the Y of the a6r2 is "2332050.631847188".

[create a7r2]
a7r2 is a area. "An area (7) in r2".
the printed name of the a7r2 is "Room 2".
Understand "Area 7 in Room 2" as a7r2.
the parent of the a7r2 is r2.
the X of the a7r2 is "548337.2000000004".
the Y of the a7r2 is "2332185.378981202".

[create a11r2]
a11r2 is a area. "An area (11) in r2".
the printed name of the a11r2 is "Room 2".
Understand "Area 11 in Room 2" as a11r2.
the parent of the a11r2 is r2.
the X of the a11r2 is "548650.8".
the Y of the a11r2 is "2332201.215267176".

[create a51r2]
a51r2 is a area. "An area (51) in r2".
the printed name of the a51r2 is "Room 2".
Understand "Area 51 in Room 2" as a51r2.
the parent of the a51r2 is r2.
the X of the a51r2 is "548690.0".
the Y of the a51r2 is "2332240.0".

[create a56r2]
a56r2 is a area. "An area (56) in r2".
the printed name of the a56r2 is "Room 2".
Understand "Area 56 in Room 2" as a56r2.
the parent of the a56r2 is r2.
the X of the a56r2 is "548298.0".
the Y of the a56r2 is "2332240.0".

[create a14r3]
a14r3 is a area. "An area (14) in r3".
the printed name of the a14r3 is "Room 3".
Understand "Area 14 in Room 3" as a14r3.
the parent of the a14r3 is r3.
the X of the a14r3 is "548550.0755098722".
the Y of the a14r3 is "2330141.8660873063".

[create a15r3]
a15r3 is a area. "An area (15) in r3".
the printed name of the a15r3 is "Room 3".
Understand "Area 15 in Room 3" as a15r3.
the parent of the a15r3 is r3.
the X of the a15r3 is "548654.875".
the Y of the a15r3 is "2330027.5439739414".

[create a23r3]
a23r3 is a area. "An area (23) in r3".
the printed name of the a23r3 is "Room 3".
Understand "Area 23 in Room 3" as a23r3.
the parent of the a23r3 is r3.
the X of the a23r3 is "548608.2988859499".
the Y of the a23r3 is "2330503.802581566".

[create a24r3]
a24r3 is a area. "An area (24) in r3".
the printed name of the a24r3 is "Room 3".
Understand "Area 24 in Room 3" as a24r3.
the parent of the a24r3 is r3.
the X of the a24r3 is "548618.3497660419".
the Y of the a24r3 is "2330363.969540283".

[create a25r3]
a25r3 is a area. "An area (25) in r3".
the printed name of the a25r3 is "Room 3".
Understand "Area 25 in Room 3" as a25r3.
the parent of the a25r3 is r3.
the X of the a25r3 is "548778.0510135791".
the Y of the a25r3 is "2330670.670295153".

[create a29r3]
a29r3 is a area. "An area (29) in r3".
the printed name of the a29r3 is "Room 3".
Understand "Area 29 in Room 3" as a29r3.
the parent of the a29r3 is r3.
the X of the a29r3 is "549292.4266762924".
the Y of the a29r3 is "2330131.6460424936".

[create a30r3]
a30r3 is a area. "An area (30) in r3".
the printed name of the a30r3 is "Room 3".
Understand "Area 30 in Room 3" as a30r3.
the parent of the a30r3 is r3.
the X of the a30r3 is "549504.2242086418".
the Y of the a30r3 is "2330260.5446214788".

[create a31r3]
a31r3 is a area. "An area (31) in r3".
the printed name of the a31r3 is "Room 3".
Understand "Area 31 in Room 3" as a31r3.
the parent of the a31r3 is r3.
the X of the a31r3 is "548902.4349820466".
the Y of the a31r3 is "2330139.8164598118".

[create a35r3]
a35r3 is a area. "An area (35) in r3".
the printed name of the a35r3 is "Room 3".
Understand "Area 35 in Room 3" as a35r3.
the parent of the a35r3 is r3.
the X of the a35r3 is "548444.125".
the Y of the a35r3 is "2330039.2083333335".

[create a41r3]
a41r3 is a area. "An area (41) in r3".
the printed name of the a41r3 is "Room 3".
Understand "Area 41 in Room 3" as a41r3.
the parent of the a41r3 is r3.
the X of the a41r3 is "549326.0".
the Y of the a41r3 is "2330000.0".

[create a42r3]
a42r3 is a area. "An area (42) in r3".
the printed name of the a42r3 is "Room 3".
Understand "Area 42 in Room 3" as a42r3.
the parent of the a42r3 is r3.
the X of the a42r3 is "549293.5".
the Y of the a42r3 is "2330127.501257862".

[create a43r3]
a43r3 is a area. "An area (43) in r3".
the printed name of the a43r3 is "Room 3".
Understand "Area 43 in Room 3" as a43r3.
the parent of the a43r3 is r3.
the X of the a43r3 is "548841.0".
the Y of the a43r3 is "2330000.0".

[create a44r3]
a44r3 is a area. "An area (44) in r3".
the printed name of the a44r3 is "Room 3".
Understand "Area 44 in Room 3" as a44r3.
the parent of the a44r3 is r3.
the X of the a44r3 is "548877.25".
the Y of the a44r3 is "2330083.659797354".

[create a45r3]
a45r3 is a area. "An area (45) in r3".
the printed name of the a45r3 is "Room 3".
Understand "Area 45 in Room 3" as a45r3.
the parent of the a45r3 is r3.
the X of the a45r3 is "549615.0".
the Y of the a45r3 is "2330159.0".

[create a46r3]
a46r3 is a area. "An area (46) in r3".
the printed name of the a46r3 is "Room 3".
Understand "Area 46 in Room 3" as a46r3.
the parent of the a46r3 is r3.
the X of the a46r3 is "549538.9294425242".
the Y of the a46r3 is "2330228.8376799403".

[create a52r3]
a52r3 is a area. "An area (52) in r3".
the printed name of the a52r3 is "Room 3".
Understand "Area 52 in Room 3" as a52r3.
the parent of the a52r3 is r3.
the X of the a52r3 is "548802.0".
the Y of the a52r3 is "2330704.0".

[create a60r3]
a60r3 is a area. "An area (60) in r3".
the printed name of the a60r3 is "Room 3".
Understand "Area 60 in Room 3" as a60r3.
the parent of the a60r3 is r3.
the X of the a60r3 is "548690.0".
the Y of the a60r3 is "2330000.0".

[create a61r3]
a61r3 is a area. "An area (61) in r3".
the printed name of the a61r3 is "Room 3".
Understand "Area 61 in Room 3" as a61r3.
the parent of the a61r3 is r3.
the X of the a61r3 is "548409.0".
the Y of the a61r3 is "2330000.0".

[create a3r4]
a3r4 is a area. "An area (3) in r4".
the printed name of the a3r4 is "Room 4".
Understand "Area 3 in Room 4" as a3r4.
the parent of the a3r4 is r4.
the X of the a3r4 is "549603.6892750249".
the Y of the a3r4 is "2332094.17983871".

[create a4r4]
a4r4 is a area. "An area (4) in r4".
the printed name of the a4r4 is "Room 4".
Understand "Area 4 in Room 4" as a4r4.
the parent of the a4r4 is r4.
the X of the a4r4 is "549712.625".
the Y of the a4r4 is "2332200.5".

[create a8r4]
a8r4 is a area. "An area (8) in r4".
the printed name of the a8r4 is "Room 4".
Understand "Area 8 in Room 4" as a8r4.
the parent of the a8r4 is r4.
the X of the a8r4 is "549494.3750000001".
the Y of the a8r4 is "2332202.0".

[create a55r4]
a55r4 is a area. "An area (55) in r4".
the printed name of the a55r4 is "Room 4".
Understand "Area 55 in Room 4" as a55r4.
the parent of the a55r4 is r4.
the X of the a55r4 is "549749.0".
the Y of the a55r4 is "2332240.0".

[create a57r4]
a57r4 is a area. "An area (57) in r4".
the printed name of the a57r4 is "Room 4".
Understand "Area 57 in Room 4" as a57r4.
the parent of the a57r4 is r4.
the X of the a57r4 is "549458.0".
the Y of the a57r4 is "2332240.0".

[create a12r5]
a12r5 is a area. "An area (12) in r5".
the printed name of the a12r5 is "Room 5".
Understand "Area 12 in Room 5" as a12r5.
the parent of the a12r5 is r5.
the X of the a12r5 is "546802.6718758782".
the Y of the a12r5 is "2330211.565848383".

[create a13r5]
a13r5 is a area. "An area (13) in r5".
the printed name of the a13r5 is "Room 5".
Understand "Area 13 in Room 5" as a13r5.
the parent of the a13r5 is r5.
the X of the a13r5 is "547141.5".
the Y of the a13r5 is "2330019.754935642".

[create a26r5]
a26r5 is a area. "An area (26) in r5".
the printed name of the a26r5 is "Room 5".
Understand "Area 26 in Room 5" as a26r5.
the parent of the a26r5 is r5.
the X of the a26r5 is "546223.4385945697".
the Y of the a26r5 is "2331111.1767984712".

[create a27r5]
a27r5 is a area. "An area (27) in r5".
the printed name of the a27r5 is "Room 5".
Understand "Area 27 in Room 5" as a27r5.
the parent of the a27r5 is r5.
the X of the a27r5 is "546653.3670987858".
the Y of the a27r5 is "2330687.7965588183".

[create a33r5]
a33r5 is a area. "An area (33) in r5".
the printed name of the a33r5 is "Room 5".
Understand "Area 33 in Room 5" as a33r5.
the parent of the a33r5 is r5.
the X of the a33r5 is "545387.606572343".
the Y of the a33r5 is "2330145.861537123".

[create a36r5]
a36r5 is a area. "An area (36) in r5".
the printed name of the a36r5 is "Room 5".
Understand "Area 36 in Room 5" as a36r5.
the parent of the a36r5 is r5.
the X of the a36r5 is "545496.625".
the Y of the a36r5 is "2330039.384615384".

[create a37r5]
a37r5 is a area. "An area (37) in r5".
the printed name of the a37r5 is "Room 5".
Understand "Area 37 in Room 5" as a37r5.
the parent of the a37r5 is r5.
the X of the a37r5 is "545278.375".
the Y of the a37r5 is "2330038.620689655".

[create a38r5]
a38r5 is a area. "An area (38) in r5".
the printed name of the a38r5 is "Room 5".
Understand "Area 38 in Room 5" as a38r5.
the parent of the a38r5 is r5.
the X of the a38r5 is "546628.1666666667".
the Y of the a38r5 is "2330036.7835051543".

[create a59r5]
a59r5 is a area. "An area (59) in r5".
the printed name of the a59r5 is "Room 5".
Understand "Area 59 in Room 5" as a59r5.
the parent of the a59r5 is r5.
the X of the a59r5 is "547180.0".
the Y of the a59r5 is "2330000.0".

[create a62r5]
a62r5 is a area. "An area (62) in r5".
the printed name of the a62r5 is "Room 5".
Understand "Area 62 in Room 5" as a62r5.
the parent of the a62r5 is r5.
the X of the a62r5 is "545533.0".
the Y of the a62r5 is "2330000.0".

[create a63r5]
a63r5 is a area. "An area (63) in r5".
the printed name of the a63r5 is "Room 5".
Understand "Area 63 in Room 5" as a63r5.
the parent of the a63r5 is r5.
the X of the a63r5 is "545242.0".
the Y of the a63r5 is "2330000.0".

[create a64r5]
a64r5 is a area. "An area (64) in r5".
the printed name of the a64r5 is "Room 5".
Understand "Area 64 in Room 5" as a64r5.
the parent of the a64r5 is r5.
the X of the a64r5 is "546591.0".
the Y of the a64r5 is "2330000.0".

[create d0]
d0 is a door. "A door between a32r1 and a5r2".
d0 is north of a32r1 and south of a5r2.

[create d2]
d2 is a door. "A door between a16r0 and a27r5".
d2 is south of a16r0 and north of a27r5.

[create d3]
d3 is a door. "A door between a28r0 and a26r5".
d3 is east of a28r0 and west of a26r5.

[create d4]
d4 is a door. "A door between a20r0 and a34r1".
d4 is south of a20r0 and north of a34r1.

[create d6]
d6 is a door. "A door between a30r3 and a3r4".
d6 is north of a30r3 and south of a3r4.

[create d8]
d8 is a door. "A door between a6r2 and a23r3".
d8 is south of a6r2 and north of a23r3.

[create d10]
d10 is a door. "A door between a16r0 and a26r5".
d10 is west of a16r0 and east of a26r5.

[create d11]
d11 is a door. "A door between a28r0 and a33r5".
d11 is south of a28r0 and north of a33r5.

southeast of a0r0 is northwest of a1r0.

northwest of a0r0 is southeast of a2r0.

northeast of a3r4 is southwest of a4r4.

east of a5r2 is west of a6r2.

northwest of a5r2 is southeast of a7r2.

northwest of a3r4 is southeast of a8r4.

southeast of a9r0 is northwest of a10r0.

northeast of a6r2 is southwest of a11r2.

southeast of a12r5 is northwest of a13r5.

southeast of a14r3 is northwest of a15r3.

north of a16r0 is south of a17r0.

east of a17r0 is west of a18r0.

south of a19r0 is north of a20r0.

north of a17r0 is south of a21r0.

northwest of a19r0 is southeast of a21r0.

south of a1r0 is north of a21r0.

west of a0r0 is east of a10r0.

northeast of a1r0 is southwest of a22r0.

south of a23r3 is north of a24r3.

northeast of a23r3 is southwest of a25r3.

southeast of a26r5 is northwest of a27r5.

south of a10r0 is north of a28r0.

northeast of a29r3 is southwest of a30r3.

southeast of a24r3 is northwest of a31r3.

north of a14r3 is south of a24r3.

north of a12r5 is south of a27r5.

southwest of a14r3 is northeast of a35r3.

southeast of a33r5 is northwest of a36r5.

southwest of a33r5 is northeast of a37r5.

southwest of a12r5 is northeast of a38r5.

west of a29r3 is east of a31r3.

southwest of a34r1 is northeast of a39r1.

southeast of a32r1 is northwest of a40r1.

north of a41r3 is south of a42r3.

northeast of a43r3 is southwest of a44r3.

northwest of a45r3 is southeast of a46r3.

east of a47r0 is west of a48r0.

southeast of a49r0 is northwest of a50r0.

northeast of a11r2 is southwest of a51r2.

northeast of a25r3 is southwest of a52r3.

south of a18r0 is north of a47r0.

northeast of a22r0 is southwest of a53r0.

northwest of a2r0 is southeast of a54r0.

northeast of a4r4 is southwest of a55r4.

northwest of a7r2 is southeast of a56r2.

northwest of a8r4 is southeast of a57r4.

northwest of a9r0 is southeast of a58r0.

southeast of a13r5 is northwest of a59r5.

southeast of a15r3 is northwest of a60r3.

southwest of a35r3 is northeast of a61r3.

southeast of a36r5 is northwest of a62r5.

southwest of a37r5 is northeast of a63r5.

southwest of a38r5 is northeast of a64r5.

southwest of a39r1 is northeast of a65r1.

southeast of a40r1 is northwest of a66r1.

south of a29r3 is north of a42r3.

southwest of a31r3 is northeast of a44r3.

west of a32r1 is east of a34r1.

southeast of a30r3 is northwest of a46r3.

west of a20r0 is east of a48r0.

west of a19r0 is east of a50r0.

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
the player is in a65r1.

the orientation of the player is 6.
