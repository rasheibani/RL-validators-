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

[create a12r0]
a12r0 is a area. "An area (12) in r0".
the printed name of the a12r0 is "Room 0".
Understand "Area 12 in Room 0" as a12r0.
the parent of the a12r0 is r0.
the X of the a12r0 is "545903.7496886395".
the Y of the a12r0 is "1680362.3173736061".

[create a15r0]
a15r0 is a area. "An area (15) in r0".
the printed name of the a15r0 is "Room 0".
Understand "Area 15 in Room 0" as a15r0.
the parent of the a15r0 is r0.
the X of the a15r0 is "545890.0086022713".
the Y of the a15r0 is "1680364.0104014394".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "545786.8816727516".
the Y of the a22r0 is "1680219.4952958587".

[create a24r0]
a24r0 is a area. "An area (24) in r0".
the printed name of the a24r0 is "Room 0".
Understand "Area 24 in Room 0" as a24r0.
the parent of the a24r0 is r0.
the X of the a24r0 is "545997.2968536067".
the Y of the a24r0 is "1680251.4209743899".

[create a73r0]
a73r0 is a area. "An area (73) in r0".
the printed name of the a73r0 is "Room 0".
Understand "Area 73 in Room 0" as a73r0.
the parent of the a73r0 is r0.
the X of the a73r0 is "545756.0".
the Y of the a73r0 is "1680172.0".

[create a75r0]
a75r0 is a area. "An area (75) in r0".
the printed name of the a75r0 is "Room 0".
Understand "Area 75 in Room 0" as a75r0.
the parent of the a75r0 is r0.
the X of the a75r0 is "546038.0".
the Y of the a75r0 is "1680200.0".

[create a13r1]
a13r1 is a area. "An area (13) in r1".
the printed name of the a13r1 is "Room 1".
Understand "Area 13 in Room 1" as a13r1.
the parent of the a13r1 is r1.
the X of the a13r1 is "546549.8708448356".
the Y of the a13r1 is "1682005.316291368".

[create a26r1]
a26r1 is a area. "An area (26) in r1".
the printed name of the a26r1 is "Room 1".
Understand "Area 26 in Room 1" as a26r1.
the parent of the a26r1 is r1.
the X of the a26r1 is "546544.5167685037".
the Y of the a26r1 is "1682173.513395759".

[create a28r1]
a28r1 is a area. "An area (28) in r1".
the printed name of the a28r1 is "Room 1".
Understand "Area 28 in Room 1" as a28r1.
the parent of the a28r1 is r1.
the X of the a28r1 is "546477.3333333333".
the Y of the a28r1 is "1682260.9694444444".

[create a39r1]
a39r1 is a area. "An area (39) in r1".
the printed name of the a39r1 is "Room 1".
Understand "Area 39 in Room 1" as a39r1.
the parent of the a39r1 is r1.
the X of the a39r1 is "546644.0".
the Y of the a39r1 is "1682304.0".

[create a40r1]
a40r1 is a area. "An area (40) in r1".
the printed name of the a40r1 is "Room 1".
Understand "Area 40 in Room 1" as a40r1.
the parent of the a40r1 is r1.
the X of the a40r1 is "546610.6666666665".
the Y of the a40r1 is "1682253.5364495006".

[create a78r1]
a78r1 is a area. "An area (78) in r1".
the printed name of the a78r1 is "Room 1".
Understand "Area 78 in Room 1" as a78r1.
the parent of the a78r1 is r1.
the X of the a78r1 is "546444.0".
the Y of the a78r1 is "1682304.0".

[create a11r2]
a11r2 is a area. "An area (11) in r2".
the printed name of the a11r2 is "Room 2".
Understand "Area 11 in Room 2" as a11r2.
the parent of the a11r2 is r2.
the X of the a11r2 is "547166.289651646".
the Y of the a11r2 is "1680360.8833195833".

[create a16r2]
a16r2 is a area. "An area (16) in r2".
the printed name of the a16r2 is "Room 2".
Understand "Area 16 in Room 2" as a16r2.
the parent of the a16r2 is r2.
the X of the a16r2 is "547144.2390740267".
the Y of the a16r2 is "1680362.4312827692".

[create a23r2]
a23r2 is a area. "An area (23) in r2".
the printed name of the a23r2 is "Room 2".
Understand "Area 23 in Room 2" as a23r2.
the parent of the a23r2 is r2.
the X of the a23r2 is "547048.7043713275".
the Y of the a23r2 is "1680222.706711225".

[create a25r2]
a25r2 is a area. "An area (25) in r2".
the printed name of the a25r2 is "Room 2".
Understand "Area 25 in Room 2" as a25r2.
the parent of the a25r2 is r2.
the X of the a25r2 is "547260.4539705466".
the Y of the a25r2 is "1680248.7643334027".

[create a74r2]
a74r2 is a area. "An area (74) in r2".
the printed name of the a74r2 is "Room 2".
Understand "Area 74 in Room 2" as a74r2.
the parent of the a74r2 is r2.
the X of the a74r2 is "547018.0".
the Y of the a74r2 is "1680172.0".

[create a76r2]
a76r2 is a area. "An area (76) in r2".
the printed name of the a76r2 is "Room 2".
Understand "Area 76 in Room 2" as a76r2.
the parent of the a76r2 is r2.
the X of the a76r2 is "547301.0".
the Y of the a76r2 is "1680200.0".

[create a2r3]
a2r3 is a area. "An area (2) in r3".
the printed name of the a2r3 is "Room 3".
Understand "Area 2 in Room 3" as a2r3.
the parent of the a2r3 is r3.
the X of the a2r3 is "547509.5496991468".
the Y of the a2r3 is "1681377.4635455795".

[create a3r3]
a3r3 is a area. "An area (3) in r3".
the printed name of the a3r3 is "Room 3".
Understand "Area 3 in Room 3" as a3r3.
the parent of the a3r3 is r3.
the X of the a3r3 is "547491.3751060646".
the Y of the a3r3 is "1681307.0663285349".

[create a4r3]
a4r3 is a area. "An area (4) in r3".
the printed name of the a4r3 is "Room 3".
Understand "Area 4 in Room 3" as a4r3.
the parent of the a4r3 is r3.
the X of the a4r3 is "547777.6894454771".
the Y of the a4r3 is "1681307.2580645164".

[create a5r3]
a5r3 is a area. "An area (5) in r3".
the printed name of the a5r3 is "Room 3".
Understand "Area 5 in Room 3" as a5r3.
the parent of the a5r3 is r3.
the X of the a5r3 is "547760.1804825792".
the Y of the a5r3 is "1681494.4914536567".

[create a6r3]
a6r3 is a area. "An area (6) in r3".
the printed name of the a6r3 is "Room 3".
Understand "Area 6 in Room 3" as a6r3.
the parent of the a6r3 is r3.
the X of the a6r3 is "547592.4174139354".
the Y of the a6r3 is "1681536.0966946804".

[create a7r3]
a7r3 is a area. "An area (7) in r3".
the printed name of the a7r3 is "Room 3".
Understand "Area 7 in Room 3" as a7r3.
the parent of the a7r3 is r3.
the X of the a7r3 is "549384.8718497385".
the Y of the a7r3 is "1682195.9455539708".

[create a8r3]
a8r3 is a area. "An area (8) in r3".
the printed name of the a8r3 is "Room 3".
Understand "Area 8 in Room 3" as a8r3.
the parent of the a8r3 is r3.
the X of the a8r3 is "547593.6987700827".
the Y of the a8r3 is "1681714.5259778565".

[create a9r3]
a9r3 is a area. "An area (9) in r3".
the printed name of the a9r3 is "Room 3".
Understand "Area 9 in Room 3" as a9r3.
the parent of the a9r3 is r3.
the X of the a9r3 is "547676.9486065779".
the Y of the a9r3 is "1682178.1064398694".

[create a10r3]
a10r3 is a area. "An area (10) in r3".
the printed name of the a10r3 is "Room 3".
Understand "Area 10 in Room 3" as a10r3.
the parent of the a10r3 is r3.
the X of the a10r3 is "549458.1150421605".
the Y of the a10r3 is "1681859.7027389589".

[create a27r3]
a27r3 is a area. "An area (27) in r3".
the printed name of the a27r3 is "Room 3".
Understand "Area 27 in Room 3" as a27r3.
the parent of the a27r3 is r3.
the X of the a27r3 is "549439.7727272728".
the Y of the a27r3 is "1682281.0".

[create a31r3]
a31r3 is a area. "An area (31) in r3".
the printed name of the a31r3 is "Room 3".
Understand "Area 31 in Room 3" as a31r3.
the parent of the a31r3 is r3.
the X of the a31r3 is "547342.4921500895".
the Y of the a31r3 is "1682260.7986347186".

[create a32r3]
a32r3 is a area. "An area (32) in r3".
the printed name of the a32r3 is "Room 3".
Understand "Area 32 in Room 3" as a32r3.
the parent of the a32r3 is r3.
the X of the a32r3 is "548006.5509290383".
the Y of the a32r3 is "1682261.0062088708".

[create a33r3]
a33r3 is a area. "An area (33) in r3".
the printed name of the a33r3 is "Room 3".
Understand "Area 33 in Room 3" as a33r3.
the parent of the a33r3 is r3.
the X of the a33r3 is "548063.0".
the Y of the a33r3 is "1682224.0".

[create a34r3]
a34r3 is a area. "An area (34) in r3".
the printed name of the a34r3 is "Room 3".
Understand "Area 34 in Room 3" as a34r3.
the parent of the a34r3 is r3.
the X of the a34r3 is "548022.6625777471".
the Y of the a34r3 is "1682242.6912577746".

[create a43r3]
a43r3 is a area. "An area (43) in r3".
the printed name of the a43r3 is "Room 3".
Understand "Area 43 in Room 3" as a43r3.
the parent of the a43r3 is r3.
the X of the a43r3 is "547295.0".
the Y of the a43r3 is "1682314.0".

[create a44r3]
a44r3 is a area. "An area (44) in r3".
the printed name of the a44r3 is "Room 3".
Understand "Area 44 in Room 3" as a44r3.
the parent of the a44r3 is r3.
the X of the a44r3 is "547325.9153780068".
the Y of the a44r3 is "1682288.1834621993".

[create a47r3]
a47r3 is a area. "An area (47) in r3".
the printed name of the a47r3 is "Room 3".
Understand "Area 47 in Room 3" as a47r3.
the parent of the a47r3 is r3.
the X of the a47r3 is "548054.0".
the Y of the a47r3 is "1682314.0".

[create a48r3]
a48r3 is a area. "An area (48) in r3".
the printed name of the a48r3 is "Room 3".
Understand "Area 48 in Room 3" as a48r3.
the parent of the a48r3 is r3.
the X of the a48r3 is "548022.9123931624".
the Y of the a48r3 is "1682288.1662393163".

[create a57r3]
a57r3 is a area. "An area (57) in r3".
the printed name of the a57r3 is "Room 3".
Understand "Area 57 in Room 3" as a57r3.
the parent of the a57r3 is r3.
the X of the a57r3 is "547682.225952813".
the Y of the a57r3 is "1681655.556261343".

[create a58r3]
a58r3 is a area. "An area (58) in r3".
the printed name of the a58r3 is "Room 3".
Understand "Area 58 in Room 3" as a58r3.
the parent of the a58r3 is r3.
the X of the a58r3 is "547688.3121597094".
the Y of the a58r3 is "1681691.061857229".

[create a59r3]
a59r3 is a area. "An area (59) in r3".
the printed name of the a59r3 is "Room 3".
Understand "Area 59 in Room 3" as a59r3.
the parent of the a59r3 is r3.
the X of the a59r3 is "549426.0".
the Y of the a59r3 is "1681788.0".

[create a60r3]
a60r3 is a area. "An area (60) in r3".
the printed name of the a60r3 is "Room 3".
Understand "Area 60 in Room 3" as a60r3.
the parent of the a60r3 is r3.
the X of the a60r3 is "549434.3098646167".
the Y of the a60r3 is "1681827.908152513".

[create a61r3]
a61r3 is a area. "An area (61) in r3".
the printed name of the a61r3 is "Room 3".
Understand "Area 61 in Room 3" as a61r3.
the parent of the a61r3 is r3.
the X of the a61r3 is "549512.0".
the Y of the a61r3 is "1681825.0".

[create a62r3]
a62r3 is a area. "An area (62) in r3".
the printed name of the a62r3 is "Room 3".
Understand "Area 62 in Room 3" as a62r3.
the parent of the a62r3 is r3.
the X of the a62r3 is "549473.9614546174".
the Y of the a62r3 is "1681854.190943322".

[create a63r3]
a63r3 is a area. "An area (63) in r3".
the printed name of the a63r3 is "Room 3".
Understand "Area 63 in Room 3" as a63r3.
the parent of the a63r3 is r3.
the X of the a63r3 is "547286.0".
the Y of the a63r3 is "1682224.0".

[create a64r3]
a64r3 is a area. "An area (64) in r3".
the printed name of the a64r3 is "Room 3".
Understand "Area 64 in Room 3" as a64r3.
the parent of the a64r3 is r3.
the X of the a64r3 is "547326.7622699386".
the Y of the a64r3 is "1682242.6487730062".

[create a77r3]
a77r3 is a area. "An area (77) in r3".
the printed name of the a77r3 is "Room 3".
Understand "Area 77 in Room 3" as a77r3.
the parent of the a77r3 is r3.
the X of the a77r3 is "549475.0".
the Y of the a77r3 is "1682304.0".

[create a0r4]
a0r4 is a area. "An area (0) in r4".
the printed name of the a0r4 is "Room 4".
Understand "Area 0 in Room 4" as a0r4.
the parent of the a0r4 is r4.
the X of the a0r4 is "549513.8477146516".
the Y of the a0r4 is "1681047.333831696".

[create a1r4]
a1r4 is a area. "An area (1) in r4".
the printed name of the a1r4 is "Room 4".
Understand "Area 1 in Room 4" as a1r4.
the parent of the a1r4 is r4.
the X of the a1r4 is "549684.8618197278".
the Y of the a1r4 is "1681094.1585884355".

[create a17r4]
a17r4 is a area. "An area (17) in r4".
the printed name of the a17r4 is "Room 4".
Understand "Area 17 in Room 4" as a17r4.
the parent of the a17r4 is r4.
the X of the a17r4 is "549406.6979495725".
the Y of the a17r4 is "1680466.6546164062".

[create a18r4]
a18r4 is a area. "An area (18) in r4".
the printed name of the a18r4 is "Room 4".
Understand "Area 18 in Room 4" as a18r4.
the parent of the a18r4 is r4.
the X of the a18r4 is "549404.2063772489".
the Y of the a18r4 is "1680968.2701607419".

[create a19r4]
a19r4 is a area. "An area (19) in r4".
the printed name of the a19r4 is "Room 4".
Understand "Area 19 in Room 4" as a19r4.
the parent of the a19r4 is r4.
the X of the a19r4 is "547796.6742984806".
the Y of the a19r4 is "1680791.579100882".

[create a20r4]
a20r4 is a area. "An area (20) in r4".
the printed name of the a20r4 is "Room 4".
Understand "Area 20 in Room 4" as a20r4.
the parent of the a20r4 is r4.
the X of the a20r4 is "549050.020066252".
the Y of the a20r4 is "1681089.22307619".

[create a21r4]
a21r4 is a area. "An area (21) in r4".
the printed name of the a21r4 is "Room 4".
Understand "Area 21 in Room 4" as a21r4.
the parent of the a21r4 is r4.
the X of the a21r4 is "548259.3005528575".
the Y of the a21r4 is "1680298.8583478578".

[create a49r4]
a49r4 is a area. "An area (49) in r4".
the printed name of the a49r4 is "Room 4".
Understand "Area 49 in Room 4" as a49r4.
the parent of the a49r4 is r4.
the X of the a49r4 is "548993.0".
the Y of the a49r4 is "1681054.0".

[create a50r4]
a50r4 is a area. "An area (50) in r4".
the printed name of the a50r4 is "Room 4".
Understand "Area 50 in Room 4" as a50r4.
the parent of the a50r4 is r4.
the X of the a50r4 is "549035.2240682533".
the Y of the a50r4 is "1681072.207678491".

[create a51r4]
a51r4 is a area. "An area (51) in r4".
the printed name of the a51r4 is "Room 4".
Understand "Area 51 in Room 4" as a51r4.
the parent of the a51r4 is r4.
the X of the a51r4 is "549739.0".
the Y of the a51r4 is "1681054.0".

[create a52r4]
a52r4 is a area. "An area (52) in r4".
the printed name of the a52r4 is "Room 4".
Understand "Area 52 in Room 4" as a52r4.
the parent of the a52r4 is r4.
the X of the a52r4 is "549711.2704191618".
the Y of the a52r4 is "1681073.6734131738".

[create a53r4]
a53r4 is a area. "An area (53) in r4".
the printed name of the a53r4 is "Room 4".
Understand "Area 53 in Room 4" as a53r4.
the parent of the a53r4 is r4.
the X of the a53r4 is "549002.0".
the Y of the a53r4 is "1681143.0".

[create a54r4]
a54r4 is a area. "An area (54) in r4".
the printed name of the a54r4 is "Room 4".
Understand "Area 54 in Room 4" as a54r4.
the parent of the a54r4 is r4.
the X of the a54r4 is "549033.0420577895".
the Y of the a54r4 is "1681117.3833874147".

[create a55r4]
a55r4 is a area. "An area (55) in r4".
the printed name of the a55r4 is "Room 4".
Understand "Area 55 in Room 4" as a55r4.
the parent of the a55r4 is r4.
the X of the a55r4 is "549730.0".
the Y of the a55r4 is "1681143.0".

[create a56r4]
a56r4 is a area. "An area (56) in r4".
the printed name of the a56r4 is "Room 4".
Understand "Area 56 in Room 4" as a56r4.
the parent of the a56r4 is r4.
the X of the a56r4 is "549698.7862838914".
the Y of the a56r4 is "1681117.3660287084".

[create a65r4]
a65r4 is a area. "An area (65) in r4".
the printed name of the a65r4 is "Room 4".
Understand "Area 65 in Room 4" as a65r4.
the parent of the a65r4 is r4.
the X of the a65r4 is "548206.0".
the Y of the a65r4 is "1680163.0".

[create a66r4]
a66r4 is a area. "An area (66) in r4".
the printed name of the a66r4 is "Room 4".
Understand "Area 66 in Room 4" as a66r4.
the parent of the a66r4 is r4.
the X of the a66r4 is "548242.0".
the Y of the a66r4 is "1680255.3867768594".

[create a67r4]
a67r4 is a area. "An area (67) in r4".
the printed name of the a67r4 is "Room 4".
Understand "Area 67 in Room 4" as a67r4.
the parent of the a67r4 is r4.
the X of the a67r4 is "549549.0".
the Y of the a67r4 is "1680369.0".

[create a68r4]
a68r4 is a area. "An area (68) in r4".
the printed name of the a68r4 is "Room 4".
Understand "Area 68 in Room 4" as a68r4.
the parent of the a68r4 is r4.
the X of the a68r4 is "549490.3376066436".
the Y of the a68r4 is "1680410.784647849".

[create a69r4]
a69r4 is a area. "An area (69) in r4".
the printed name of the a69r4 is "Room 4".
Understand "Area 69 in Room 4" as a69r4.
the parent of the a69r4 is r4.
the X of the a69r4 is "547616.0".
the Y of the a69r4 is "1680713.0".

[create a70r4]
a70r4 is a area. "An area (70) in r4".
the printed name of the a70r4 is "Room 4".
Understand "Area 70 in Room 4" as a70r4.
the parent of the a70r4 is r4.
the X of the a70r4 is "547700.2818622056".
the Y of the a70r4 is "1680749.642857143".

[create a71r4]
a71r4 is a area. "An area (71) in r4".
the printed name of the a71r4 is "Room 4".
Understand "Area 71 in Room 4" as a71r4.
the parent of the a71r4 is r4.
the X of the a71r4 is "549598.0".
the Y of the a71r4 is "1681017.0".

[create a72r4]
a72r4 is a area. "An area (72) in r4".
the printed name of the a72r4 is "Room 4".
Understand "Area 72 in Room 4" as a72r4.
the parent of the a72r4 is r4.
the X of the a72r4 is "549516.1666666665".
the Y of the a72r4 is "1681045.8333333335".

[create a14r5]
a14r5 is a area. "An area (14) in r5".
the printed name of the a14r5 is "Room 5".
Understand "Area 14 in Room 5" as a14r5.
the parent of the a14r5 is r5.
the X of the a14r5 is "545405.79770735".
the Y of the a14r5 is "1682144.567768038".

[create a29r5]
a29r5 is a area. "An area (29) in r5".
the printed name of the a29r5 is "Room 5".
Understand "Area 29 in Room 5" as a29r5.
the parent of the a29r5 is r5.
the X of the a29r5 is "545050.2813718207".
the Y of the a29r5 is "1682261.2845894436".

[create a30r5]
a30r5 is a area. "An area (30) in r5".
the printed name of the a30r5 is "Room 5".
Understand "Area 30 in Room 5" as a30r5.
the parent of the a30r5 is r5.
the X of the a30r5 is "545779.1653756485".
the Y of the a30r5 is "1682261.2658517451".

[create a35r5]
a35r5 is a area. "An area (35) in r5".
the printed name of the a35r5 is "Room 5".
Understand "Area 35 in Room 5" as a35r5.
the parent of the a35r5 is r5.
the X of the a35r5 is "544994.0".
the Y of the a35r5 is "1682224.0".

[create a36r5]
a36r5 is a area. "An area (36) in r5".
the printed name of the a36r5 is "Room 5".
Understand "Area 36 in Room 5" as a36r5.
the parent of the a36r5 is r5.
the X of the a36r5 is "545033.4941364606".
the Y of the a36r5 is "1682242.7755863538".

[create a37r5]
a37r5 is a area. "An area (37) in r5".
the printed name of the a37r5 is "Room 5".
Understand "Area 37 in Room 5" as a37r5.
the parent of the a37r5 is r5.
the X of the a37r5 is "545836.0".
the Y of the a37r5 is "1682224.0".

[create a38r5]
a38r5 is a area. "An area (38) in r5".
the printed name of the a38r5 is "Room 5".
Understand "Area 38 in Room 5" as a38r5.
the parent of the a38r5 is r5.
the X of the a38r5 is "545795.6686930091".
the Y of the a38r5 is "1682242.2965214453".

[create a41r5]
a41r5 is a area. "An area (41) in r5".
the printed name of the a41r5 is "Room 5".
Understand "Area 41 in Room 5" as a41r5.
the parent of the a41r5 is r5.
the X of the a41r5 is "545826.0".
the Y of the a41r5 is "1682314.0".

[create a42r5]
a42r5 is a area. "An area (42) in r5".
the printed name of the a42r5 is "Room 5".
Understand "Area 42 in Room 5" as a42r5.
the parent of the a42r5 is r5.
the X of the a42r5 is "545795.1050372067".
the Y of the a42r5 is "1682287.7894485786".

[create a45r5]
a45r5 is a area. "An area (45) in r5".
the printed name of the a45r5 is "Room 5".
Understand "Area 45 in Room 5" as a45r5.
the parent of the a45r5 is r5.
the X of the a45r5 is "545003.0".
the Y of the a45r5 is "1682314.0".

[create a46r5]
a46r5 is a area. "An area (46) in r5".
the printed name of the a46r5 is "Room 5".
Understand "Area 46 in Room 5" as a46r5.
the parent of the a46r5 is r5.
the X of the a46r5 is "545034.0876068375".
the Y of the a46r5 is "1682288.1662393163".

[create d0]
d0 is a door. "A door between a11r2 and a3r3".
d0 is north of a11r2 and south of a3r3.

[create d2]
d2 is a door. "A door between a12r0 and a13r1".
d2 is north of a12r0 and south of a13r1.

[create d4]
d4 is a door. "A door between a15r0 and a14r5".
d4 is north of a15r0 and south of a14r5.

[create d6]
d6 is a door. "A door between a13r1 and a16r2".
d6 is southeast of a13r1 and northwest of a16r2.

[create d8]
d8 is a door. "A door between a4r3 and a19r4".
d8 is south of a4r3 and north of a19r4.

east of a0r4 is west of a1r4.

south of a2r3 is north of a3r3.

north of a4r3 is south of a5r3.

northeast of a2r3 is southwest of a6r3.

northeast of a5r3 is southwest of a7r3.

north of a6r3 is south of a8r3.

north of a8r3 is south of a9r3.

south of a7r3 is north of a10r3.

north of a17r4 is south of a18r4.

west of a18r4 is east of a20r4.

southwest of a0r4 is northeast of a18r4.

west of a17r4 is east of a21r4.

southwest of a15r0 is northeast of a22r0.

southwest of a16r2 is northeast of a23r2.

southeast of a19r4 is northwest of a21r4.

southeast of a12r0 is northwest of a24r0.

southeast of a11r2 is northwest of a25r2.

north of a13r1 is south of a26r1.

northeast of a7r3 is southwest of a27r3.

northwest of a26r1 is southeast of a28r1.

west of a14r5 is east of a29r5.

east of a14r5 is west of a30r5.

west of a9r3 is east of a31r3.

east of a9r3 is west of a32r3.

northwest of a33r3 is southeast of a34r3.

northeast of a35r5 is southwest of a36r5.

northwest of a37r5 is southeast of a38r5.

southwest of a39r1 is northeast of a40r1.

southwest of a41r5 is northeast of a42r5.

southeast of a43r3 is northwest of a44r3.

southeast of a45r5 is northwest of a46r5.

southwest of a47r3 is northeast of a48r3.

northeast of a49r4 is southwest of a50r4.

northwest of a51r4 is southeast of a52r4.

southeast of a53r4 is northwest of a54r4.

southwest of a55r4 is northeast of a56r4.

north of a57r3 is south of a58r3.

north of a59r3 is south of a60r3.

northwest of a61r3 is southeast of a62r3.

northeast of a63r3 is southwest of a64r3.

north of a65r4 is south of a66r4.

northwest of a67r4 is southeast of a68r4.

northeast of a69r4 is southwest of a70r4.

west of a71r4 is east of a72r4.

southwest of a22r0 is northeast of a73r0.

southwest of a23r2 is northeast of a74r2.

southeast of a24r0 is northwest of a75r0.

southeast of a25r2 is northwest of a76r2.

northeast of a27r3 is southwest of a77r3.

northwest of a28r1 is southeast of a78r1.

west of a12r0 is east of a15r0.

southeast of a17r4 is northwest of a68r4.

southwest of a19r4 is northeast of a70r4.

southeast of a0r4 is northwest of a72r4.

southwest of a20r4 is northeast of a50r4.

southeast of a1r4 is northwest of a52r4.

northwest of a20r4 is southeast of a54r4.

northeast of a1r4 is southwest of a56r4.

west of a5r3 is east of a6r3.

south of a21r4 is north of a66r4.

west of a11r2 is east of a16r2.

northeast of a26r1 is southwest of a40r1.

northeast of a30r5 is southwest of a42r5.

northwest of a31r3 is southeast of a44r3.

northwest of a29r5 is southeast of a46r5.

northeast of a32r3 is southwest of a48r3.

east of a8r3 is west of a58r3.

southwest of a10r3 is northeast of a60r3.

east of a10r3 is west of a62r3.

southwest of a31r3 is northeast of a64r3.

southeast of a32r3 is northwest of a34r3.

southwest of a29r5 is northeast of a36r5.

southeast of a30r5 is northwest of a38r5.

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
the player is in a67r4.

the orientation of the player is 6.
