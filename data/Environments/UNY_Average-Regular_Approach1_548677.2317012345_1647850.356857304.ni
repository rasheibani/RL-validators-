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

[create r6]
r6 is a indoor_room. "Room 6".

[create a2r0]
a2r0 is a area. "An area (2) in r0".
the printed name of the a2r0 is "Room 0".
Understand "Area 2 in Room 0" as a2r0.
the parent of the a2r0 is r0.
the X of the a2r0 is "547108.082623469".
the Y of the a2r0 is "1647854.1734008307".

[create a31r0]
a31r0 is a area. "An area (31) in r0".
the printed name of the a31r0 is "Room 0".
Understand "Area 31 in Room 0" as a31r0.
the parent of the a31r0 is r0.
the X of the a31r0 is "546559.508099352".
the Y of the a31r0 is "1648064.2282217424".

[create a32r0]
a32r0 is a area. "An area (32) in r0".
the printed name of the a32r0 is "Room 0".
Understand "Area 32 in Room 0" as a32r0.
the parent of the a32r0 is r0.
the X of the a32r0 is "546779.4825390012".
the Y of the a32r0 is "1648040.537784457".

[create a33r0]
a33r0 is a area. "An area (33) in r0".
the printed name of the a33r0 is "Room 0".
Understand "Area 33 in Room 0" as a33r0.
the parent of the a33r0 is r0.
the X of the a33r0 is "547052.2120426875".
the Y of the a33r0 is "1647852.5478702197".

[create a36r0]
a36r0 is a area. "An area (36) in r0".
the printed name of the a36r0 is "Room 0".
Understand "Area 36 in Room 0" as a36r0.
the parent of the a36r0 is r0.
the X of the a36r0 is "547077.0212266059".
the Y of the a36r0 is "1647859.0507179871".

[create a37r0]
a37r0 is a area. "An area (37) in r0".
the printed name of the a37r0 is "Room 0".
Understand "Area 37 in Room 0" as a37r0.
the parent of the a37r0 is r0.
the X of the a37r0 is "547232.1623504465".
the Y of the a37r0 is "1648061.2608178095".

[create a41r0]
a41r0 is a area. "An area (41) in r0".
the printed name of the a41r0 is "Room 0".
Understand "Area 41 in Room 0" as a41r0.
the parent of the a41r0 is r0.
the X of the a41r0 is "547005.9760554549".
the Y of the a41r0 is "1647169.1666475174".

[create a48r0]
a48r0 is a area. "An area (48) in r0".
the printed name of the a48r0 is "Room 0".
Understand "Area 48 in Room 0" as a48r0.
the parent of the a48r0 is r0.
the X of the a48r0 is "546514.0".
the Y of the a48r0 is "1648114.0".

[create a49r0]
a49r0 is a area. "An area (49) in r0".
the printed name of the a49r0 is "Room 0".
Understand "Area 49 in Room 0" as a49r0.
the parent of the a49r0 is r0.
the X of the a49r0 is "546545.0876068377".
the Y of the a49r0 is "1648088.1662393163".

[create a50r0]
a50r0 is a area. "An area (50) in r0".
the printed name of the a50r0 is "Room 0".
Understand "Area 50 in Room 0" as a50r0.
the parent of the a50r0 is r0.
the X of the a50r0 is "546741.0".
the Y of the a50r0 is "1648114.0".

[create a51r0]
a51r0 is a area. "An area (51) in r0".
the printed name of the a51r0 is "Room 0".
Understand "Area 51 in Room 0" as a51r0.
the parent of the a51r0 is r0.
the X of the a51r0 is "546772.8264604809".
the Y of the a51r0 is "1648105.767353952".

[create a62r0]
a62r0 is a area. "An area (62) in r0".
the printed name of the a62r0 is "Room 0".
Understand "Area 62 in Room 0" as a62r0.
the parent of the a62r0 is r0.
the X of the a62r0 is "546505.0".
the Y of the a62r0 is "1648024.0".

[create a63r0]
a63r0 is a area. "An area (63) in r0".
the printed name of the a63r0 is "Room 0".
Understand "Area 63 in Room 0" as a63r0.
the parent of the a63r0 is r0.
the X of the a63r0 is "546532.2059471365".
the Y of the a63r0 is "1648044.0044052864".

[create a68r0]
a68r0 is a area. "An area (68) in r0".
the printed name of the a68r0 is "Room 0".
Understand "Area 68 in Room 0" as a68r0.
the parent of the a68r0 is r0.
the X of the a68r0 is "547279.0".
the Y of the a68r0 is "1648114.0".

[create a69r0]
a69r0 is a area. "An area (69) in r0".
the printed name of the a69r0 is "Room 0".
Understand "Area 69 in Room 0" as a69r0.
the parent of the a69r0 is r0.
the X of the a69r0 is "547248.1050372066".
the Y of the a69r0 is "1648087.7894485788".

[create a3r1]
a3r1 is a area. "An area (3) in r1".
the printed name of the a3r1 is "Room 1".
Understand "Area 3 in Room 1" as a3r1.
the parent of the a3r1 is r1.
the X of the a3r1 is "548513.9405397375".
the Y of the a3r1 is "1646220.819968459".

[create a16r1]
a16r1 is a area. "An area (16) in r1".
the printed name of the a16r1 is "Room 1".
Understand "Area 16 in Room 1" as a16r1.
the parent of the a16r1 is r1.
the X of the a16r1 is "548542.9623993274".
the Y of the a16r1 is "1646142.03653799".

[create a17r1]
a17r1 is a area. "An area (17) in r1".
the printed name of the a17r1 is "Room 1".
Understand "Area 17 in Room 1" as a17r1.
the parent of the a17r1 is r1.
the X of the a17r1 is "548678.8020768752".
the Y of the a17r1 is "1646026.792467452".

[create a20r1]
a20r1 is a area. "An area (20) in r1".
the printed name of the a20r1 is "Room 1".
Understand "Area 20 in Room 1" as a20r1.
the parent of the a20r1 is r1.
the X of the a20r1 is "548490.6571255578".
the Y of the a20r1 is "1646037.8159683133".

[create a96r1]
a96r1 is a area. "An area (96) in r1".
the printed name of the a96r1 is "Room 1".
Understand "Area 96 in Room 1" as a96r1.
the parent of the a96r1 is r1.
the X of the a96r1 is "548713.0".
the Y of the a96r1 is "1646000.0".

[create a97r1]
a97r1 is a area. "An area (97) in r1".
the printed name of the a97r1 is "Room 1".
Understand "Area 97 in Room 1" as a97r1.
the parent of the a97r1 is r1.
the X of the a97r1 is "548467.0".
the Y of the a97r1 is "1645972.0".

[create a4r2]
a4r2 is a area. "An area (4) in r2".
the printed name of the a4r2 is "Room 2".
Understand "Area 4 in Room 2" as a4r2.
the parent of the a4r2 is r2.
the X of the a4r2 is "548677.2317012345".
the Y of the a4r2 is "1647850.356857304".

[create a26r2]
a26r2 is a area. "An area (26) in r2".
the printed name of the a26r2 is "Room 2".
Understand "Area 26 in Room 2" as a26r2.
the parent of the a26r2 is r2.
the X of the a26r2 is "548232.0469750918".
the Y of the a26r2 is "1648063.1459074754".

[create a27r2]
a27r2 is a area. "An area (27) in r2".
the printed name of the a27r2 is "Room 2".
Understand "Area 27 in Room 2" as a27r2.
the parent of the a27r2 is r2.
the X of the a27r2 is "548327.6699477624".
the Y of the a27r2 is "1648055.5047854779".

[create a28r2]
a28r2 is a area. "An area (28) in r2".
the printed name of the a28r2 is "Room 2".
Understand "Area 28 in Room 2" as a28r2.
the parent of the a28r2 is r2.
the X of the a28r2 is "548915.6858001074".
the Y of the a28r2 is "1648048.553656896".

[create a29r2]
a29r2 is a area. "An area (29) in r2".
the printed name of the a29r2 is "Room 2".
Understand "Area 29 in Room 2" as a29r2.
the parent of the a29r2 is r2.
the X of the a29r2 is "549089.3816789301".
the Y of the a29r2 is "1648061.096078422".

[create a30r2]
a30r2 is a area. "An area (30) in r2".
the printed name of the a30r2 is "Room 2".
Understand "Area 30 in Room 2" as a30r2.
the parent of the a30r2 is r2.
the X of the a30r2 is "548668.8798986156".
the Y of the a30r2 is "1647903.0325599532".

[create a52r2]
a52r2 is a area. "An area (52) in r2".
the printed name of the a52r2 is "Room 2".
Understand "Area 52 in Room 2" as a52r2.
the parent of the a52r2 is r2.
the X of the a52r2 is "548953.9384615384".
the Y of the a52r2 is "1648104.6153846155".

[create a53r2]
a53r2 is a area. "An area (53) in r2".
the printed name of the a53r2 is "Room 2".
Understand "Area 53 in Room 2" as a53r2.
the parent of the a53r2 is r2.
the X of the a53r2 is "548920.9900460225".
the Y of the a53r2 is "1648106.0597738328".

[create a54r2]
a54r2 is a area. "An area (54) in r2".
the printed name of the a54r2 is "Room 2".
Understand "Area 54 in Room 2" as a54r2.
the parent of the a54r2 is r2.
the X of the a54r2 is "548294.0".
the Y of the a54r2 is "1648114.0".

[create a55r2]
a55r2 is a area. "An area (55) in r2".
the printed name of the a55r2 is "Room 2".
Understand "Area 55 in Room 2" as a55r2.
the parent of the a55r2 is r2.
the X of the a55r2 is "548325.9445389167".
the Y of the a55r2 is "1648108.0462677581".

[create a64r2]
a64r2 is a area. "An area (64) in r2".
the printed name of the a64r2 is "Room 2".
Understand "Area 64 in Room 2" as a64r2.
the parent of the a64r2 is r2.
the X of the a64r2 is "548188.0".
the Y of the a64r2 is "1648114.0".

[create a65r2]
a65r2 is a area. "An area (65) in r2".
the printed name of the a65r2 is "Room 2".
Understand "Area 65 in Room 2" as a65r2.
the parent of the a65r2 is r2.
the X of the a65r2 is "548213.1265277929".
the Y of the a65r2 is "1648088.7623472204".

[create a66r2]
a66r2 is a area. "An area (66) in r2".
the printed name of the a66r2 is "Room 2".
Understand "Area 66 in Room 2" as a66r2.
the parent of the a66r2 is r2.
the X of the a66r2 is "549136.0".
the Y of the a66r2 is "1648114.0".

[create a67r2]
a67r2 is a area. "An area (67) in r2".
the printed name of the a67r2 is "Room 2".
Understand "Area 67 in Room 2" as a67r2.
the parent of the a67r2 is r2.
the X of the a67r2 is "549106.8971926365".
the Y of the a67r2 is "1648088.3647192637".

[create a82r2]
a82r2 is a area. "An area (82) in r2".
the printed name of the a82r2 is "Room 2".
Understand "Area 82 in Room 2" as a82r2.
the parent of the a82r2 is r2.
the X of the a82r2 is "548179.0".
the Y of the a82r2 is "1648024.0".

[create a83r2]
a83r2 is a area. "An area (83) in r2".
the printed name of the a83r2 is "Room 2".
Understand "Area 83 in Room 2" as a83r2.
the parent of the a83r2 is r2.
the X of the a83r2 is "548206.2059471365".
the Y of the a83r2 is "1648044.0044052864".

[create a84r2]
a84r2 is a area. "An area (84) in r2".
the printed name of the a84r2 is "Room 2".
Understand "Area 84 in Room 2" as a84r2.
the parent of the a84r2 is r2.
the X of the a84r2 is "549145.0".
the Y of the a84r2 is "1648024.0".

[create a85r2]
a85r2 is a area. "An area (85) in r2".
the printed name of the a85r2 is "Room 2".
Understand "Area 85 in Room 2" as a85r2.
the parent of the a85r2 is r2.
the X of the a85r2 is "549105.9448424068".
the Y of the a85r2 is "1648042.8194842406".

[create a0r3]
a0r3 is a area. "An area (0) in r3".
the printed name of the a0r3 is "Room 3".
Understand "Area 0 in Room 3" as a0r3.
the parent of the a0r3 is r3.
the X of the a0r3 is "549275.1730804583".
the Y of the a0r3 is "1646859.7005024815".

[create a1r3]
a1r3 is a area. "An area (1) in r3".
the printed name of the a1r3 is "Room 3".
Understand "Area 1 in Room 3" as a1r3.
the parent of the a1r3 is r3.
the X of the a1r3 is "549280.0".
the Y of the a1r3 is "1646168.5778301887".

[create a22r3]
a22r3 is a area. "An area (22) in r3".
the printed name of the a22r3 is "Room 3".
Understand "Area 22 in Room 3" as a22r3.
the parent of the a22r3 is r3.
the X of the a22r3 is "549646.7290448342".
the Y of the a22r3 is "1646045.062703054".

[create a23r3]
a23r3 is a area. "An area (23) in r3".
the printed name of the a23r3 is "Room 3".
Understand "Area 23 in Room 3" as a23r3.
the parent of the a23r3 is r3.
the X of the a23r3 is "548898.3945333334".
the Y of the a23r3 is "1646040.2686666667".

[create a72r3]
a72r3 is a area. "An area (72) in r3".
the printed name of the a72r3 is "Room 3".
Understand "Area 72 in Room 3" as a72r3.
the parent of the a72r3 is r3.
the X of the a72r3 is "548844.0".
the Y of the a72r3 is "1646080.0".

[create a73r3]
a73r3 is a area. "An area (73) in r3".
the printed name of the a73r3 is "Room 3".
Understand "Area 73 in Room 3" as a73r3.
the parent of the a73r3 is r3.
the X of the a73r3 is "548874.8730024241".
the Y of the a73r3 is "1646060.6444609193".

[create a74r3]
a74r3 is a area. "An area (74) in r3".
the printed name of the a74r3 is "Room 3".
Understand "Area 74 in Room 3" as a74r3.
the parent of the a74r3 is r3.
the X of the a74r3 is "549695.0".
the Y of the a74r3 is "1645991.0".

[create a75r3]
a75r3 is a area. "An area (75) in r3".
the printed name of the a75r3 is "Room 3".
Understand "Area 75 in Room 3" as a75r3.
the parent of the a75r3 is r3.
the X of the a75r3 is "549663.7862838916".
the Y of the a75r3 is "1646016.633971292".

[create a76r3]
a76r3 is a area. "An area (76) in r3".
the printed name of the a76r3 is "Room 3".
Understand "Area 76 in Room 3" as a76r3.
the parent of the a76r3 is r3.
the X of the a76r3 is "549704.0".
the Y of the a76r3 is "1646080.0".

[create a77r3]
a77r3 is a area. "An area (77) in r3".
the printed name of the a77r3 is "Room 3".
Understand "Area 77 in Room 3" as a77r3.
the parent of the a77r3 is r3.
the X of the a77r3 is "549660.9383292026".
the Y of the a77r3 is "1646061.87702289".

[create a90r3]
a90r3 is a area. "An area (90) in r3".
the printed name of the a90r3 is "Room 3".
Understand "Area 90 in Room 3" as a90r3.
the parent of the a90r3 is r3.
the X of the a90r3 is "548853.0".
the Y of the a90r3 is "1645991.0".

[create a91r3]
a91r3 is a area. "An area (91) in r3".
the printed name of the a91r3 is "Room 3".
Understand "Area 91 in Room 3" as a91r3.
the parent of the a91r3 is r3.
the X of the a91r3 is "548884.2137161084".
the Y of the a91r3 is "1646016.633971292".

[create a24r4]
a24r4 is a area. "An area (24) in r4".
the printed name of the a24r4 is "Room 4".
Understand "Area 24 in Room 4" as a24r4.
the parent of the a24r4 is r4.
the X of the a24r4 is "549910.3793965107".
the Y of the a24r4 is "1647974.0644405335".

[create a25r4]
a25r4 is a area. "An area (25) in r4".
the printed name of the a25r4 is "Room 4".
Understand "Area 25 in Room 4" as a25r4.
the parent of the a25r4 is r4.
the X of the a25r4 is "550231.305100284".
the Y of the a25r4 is "1648061.4983535402".

[create a38r4]
a38r4 is a area. "An area (38) in r4".
the printed name of the a38r4 is "Room 4".
Understand "Area 38 in Room 4" as a38r4.
the parent of the a38r4 is r4.
the X of the a38r4 is "549566.7072666291".
the Y of the a38r4 is "1648061.4777749956".

[create a42r4]
a42r4 is a area. "An area (42) in r4".
the printed name of the a42r4 is "Room 4".
Understand "Area 42 in Room 4" as a42r4.
the parent of the a42r4 is r4.
the X of the a42r4 is "550278.0".
the Y of the a42r4 is "1648114.0".

[create a43r4]
a43r4 is a area. "An area (43) in r4".
the printed name of the a43r4 is "Room 4".
Understand "Area 43 in Room 4" as a43r4.
the parent of the a43r4 is r4.
the X of the a43r4 is "550247.1050372066".
the Y of the a43r4 is "1648087.7894485786".

[create a58r4]
a58r4 is a area. "An area (58) in r4".
the printed name of the a58r4 is "Room 4".
Understand "Area 58 in Room 4" as a58r4.
the parent of the a58r4 is r4.
the X of the a58r4 is "549510.0".
the Y of the a58r4 is "1648024.0".

[create a59r4]
a59r4 is a area. "An area (59) in r4".
the printed name of the a59r4 is "Room 4".
Understand "Area 59 in Room 4" as a59r4.
the parent of the a59r4 is r4.
the X of the a59r4 is "549548.7029286583".
the Y of the a59r4 is "1648042.4774523713".

[create a60r4]
a60r4 is a area. "An area (60) in r4".
the printed name of the a60r4 is "Room 4".
Understand "Area 60 in Room 4" as a60r4.
the parent of the a60r4 is r4.
the X of the a60r4 is "550288.0".
the Y of the a60r4 is "1648024.0".

[create a61r4]
a61r4 is a area. "An area (61) in r4".
the printed name of the a61r4 is "Room 4".
Understand "Area 61 in Room 4" as a61r4.
the parent of the a61r4 is r4.
the X of the a61r4 is "550248.9982014389".
the Y of the a61r4 is "1648042.4442446046".

[create a70r4]
a70r4 is a area. "An area (70) in r4".
the printed name of the a70r4 is "Room 4".
Understand "Area 70 in Room 4" as a70r4.
the parent of the a70r4 is r4.
the X of the a70r4 is "549520.0".
the Y of the a70r4 is "1648114.0".

[create a71r4]
a71r4 is a area. "An area (71) in r4".
the printed name of the a71r4 is "Room 4".
Understand "Area 71 in Room 4" as a71r4.
the parent of the a71r4 is r4.
the X of the a71r4 is "549550.8949627934".
the Y of the a71r4 is "1648087.7894485786".

[create a5r5]
a5r5 is a area. "An area (5) in r5".
the printed name of the a5r5 is "Room 5".
Understand "Area 5 in Room 5" as a5r5.
the parent of the a5r5 is r5.
the X of the a5r5 is "547101.2770075102".
the Y of the a5r5 is "1646424.0".

[create a6r5]
a6r5 is a area. "An area (6) in r5".
the printed name of the a6r5 is "Room 5".
Understand "Area 6 in Room 5" as a6r5.
the parent of the a6r5 is r5.
the X of the a6r5 is "547096.902173913".
the Y of the a6r5 is "1646492.0".

[create a8r5]
a8r5 is a area. "An area (8) in r5".
the printed name of the a8r5 is "Room 5".
Understand "Area 8 in Room 5" as a8r5.
the parent of the a8r5 is r5.
the X of the a8r5 is "545513.8224172434".
the Y of the a8r5 is "1646357.5305011154".

[create a9r5]
a9r5 is a area. "An area (9) in r5".
the printed name of the a9r5 is "Room 5".
Understand "Area 9 in Room 5" as a9r5.
the parent of the a9r5 is r5.
the X of the a9r5 is "547096.8597571332".
the Y of the a9r5 is "1646560.0".

[create a10r5]
a10r5 is a area. "An area (10) in r5".
the printed name of the a10r5 is "Room 5".
Understand "Area 10 in Room 5" as a10r5.
the parent of the a10r5 is r5.
the X of the a10r5 is "546892.6381731406".
the Y of the a10r5 is "1646581.6465456714".

[create a11r5]
a11r5 is a area. "An area (11) in r5".
the printed name of the a11r5 is "Room 5".
Understand "Area 11 in Room 5" as a11r5.
the parent of the a11r5 is r5.
the X of the a11r5 is "546885.5278947368".
the Y of the a11r5 is "1646509.7793026315".

[create a12r5]
a12r5 is a area. "An area (12) in r5".
the printed name of the a12r5 is "Room 5".
Understand "Area 12 in Room 5" as a12r5.
the parent of the a12r5 is r5.
the X of the a12r5 is "547096.93359375".
the Y of the a12r5 is "1646628.0".

[create a13r5]
a13r5 is a area. "An area (13) in r5".
the printed name of the a13r5 is "Room 5".
Understand "Area 13 in Room 5" as a13r5.
the parent of the a13r5 is r5.
the X of the a13r5 is "546733.6244433795".
the Y of the a13r5 is "1646039.791415747".

[create a14r5]
a14r5 is a area. "An area (14) in r5".
the printed name of the a14r5 is "Room 5".
Understand "Area 14 in Room 5" as a14r5.
the parent of the a14r5 is r5.
the X of the a14r5 is "547104.4387151967".
the Y of the a14r5 is "1646130.6903873074".

[create a15r5]
a15r5 is a area. "An area (15) in r5".
the printed name of the a15r5 is "Room 5".
Understand "Area 15 in Room 5" as a15r5.
the parent of the a15r5 is r5.
the X of the a15r5 is "546540.3232486374".
the Y of the a15r5 is "1646084.4428030436".

[create a18r5]
a18r5 is a area. "An area (18) in r5".
the printed name of the a18r5 is "Room 5".
Understand "Area 18 in Room 5" as a18r5.
the parent of the a18r5 is r5.
the X of the a18r5 is "546620.5214740627".
the Y of the a18r5 is "1646143.5556077815".

[create a19r5]
a19r5 is a area. "An area (19) in r5".
the printed name of the a19r5 is "Room 5".
Understand "Area 19 in Room 5" as a19r5.
the parent of the a19r5 is r5.
the X of the a19r5 is "546872.493093392".
the Y of the a19r5 is "1646448.1898662767".

[create a21r5]
a21r5 is a area. "An area (21) in r5".
the printed name of the a21r5 is "Room 5".
Understand "Area 21 in Room 5" as a21r5.
the parent of the a21r5 is r5.
the X of the a21r5 is "547460.7586021507".
the Y of the a21r5 is "1646040.0134408604".

[create a39r5]
a39r5 is a area. "An area (39) in r5".
the printed name of the a39r5 is "Room 5".
Understand "Area 39 in Room 5" as a39r5.
the parent of the a39r5 is r5.
the X of the a39r5 is "546899.689081676".
the Y of the a39r5 is "1646653.5152729582".

[create a40r5]
a40r5 is a area. "An area (40) in r5".
the printed name of the a40r5 is "Room 5".
Understand "Area 40 in Room 5" as a40r5.
the parent of the a40r5 is r5.
the X of the a40r5 is "547097.1236837637".
the Y of the a40r5 is "1646696.0".

[create a78r5]
a78r5 is a area. "An area (78) in r5".
the printed name of the a78r5 is "Room 5".
Understand "Area 78 in Room 5" as a78r5.
the parent of the a78r5 is r5.
the X of the a78r5 is "546594.0".
the Y of the a78r5 is "1645963.0".

[create a79r5]
a79r5 is a area. "An area (79) in r5".
the printed name of the a79r5 is "Room 5".
Understand "Area 79 in Room 5" as a79r5.
the parent of the a79r5 is r5.
the X of the a79r5 is "546558.4166666666".
the Y of the a79r5 is "1646042.5804680437".

[create a80r5]
a80r5 is a area. "An area (80) in r5".
the printed name of the a80r5 is "Room 5".
Understand "Area 80 in Room 5" as a80r5.
the parent of the a80r5 is r5.
the X of the a80r5 is "545347.0".
the Y of the a80r5 is "1646273.0".

[create a81r5]
a81r5 is a area. "An area (81) in r5".
the printed name of the a81r5 is "Room 5".
Understand "Area 81 in Room 5" as a81r5.
the parent of the a81r5 is r5.
the X of the a81r5 is "545417.9942396786".
the Y of the a81r5 is "1646311.6157052463".

[create a88r5]
a88r5 is a area. "An area (88) in r5".
the printed name of the a88r5 is "Room 5".
Understand "Area 88 in Room 5" as a88r5.
the parent of the a88r5 is r5.
the X of the a88r5 is "547506.0".
the Y of the a88r5 is "1645991.0".

[create a89r5]
a89r5 is a area. "An area (89) in r5".
the printed name of the a89r5 is "Room 5".
Understand "Area 89 in Room 5" as a89r5.
the parent of the a89r5 is r5.
the X of the a89r5 is "547474.7862838916".
the Y of the a89r5 is "1646016.633971292".

[create a92r5]
a92r5 is a area. "An area (92) in r5".
the printed name of the a92r5 is "Room 5".
Understand "Area 92 in Room 5" as a92r5.
the parent of the a92r5 is r5.
the X of the a92r5 is "546695.0".
the Y of the a92r5 is "1645991.0".

[create a93r5]
a93r5 is a area. "An area (93) in r5".
the printed name of the a93r5 is "Room 5".
Understand "Area 93 in Room 5" as a93r5.
the parent of the a93r5 is r5.
the X of the a93r5 is "546725.3518685468".
the Y of the a93r5 is "1646031.4173374595".

[create a94r5]
a94r5 is a area. "An area (94) in r5".
the printed name of the a94r5 is "Room 5".
Understand "Area 94 in Room 5" as a94r5.
the parent of the a94r5 is r5.
the X of the a94r5 is "547515.0".
the Y of the a94r5 is "1646080.0".

[create a95r5]
a95r5 is a area. "An area (95) in r5".
the printed name of the a95r5 is "Room 5".
Understand "Area 95 in Room 5" as a95r5.
the parent of the a95r5 is r5.
the X of the a95r5 is "547487.8491587283".
the Y of the a95r5 is "1646060.2680626006".

[create a7r6]
a7r6 is a area. "An area (7) in r6".
the printed name of the a7r6 is "Room 6".
Understand "Area 7 in Room 6" as a7r6.
the parent of the a7r6 is r6.
the X of the a7r6 is "545485.0".
the Y of the a7r6 is "1647937.7083333333".

[create a34r6]
a34r6 is a area. "An area (34) in r6".
the printed name of the a34r6 is "Room 6".
Understand "Area 34 in Room 6" as a34r6.
the parent of the a34r6 is r6.
the X of the a34r6 is "545839.8606168446".
the Y of the a34r6 is "1648059.860290629".

[create a35r6]
a35r6 is a area. "An area (35) in r6".
the printed name of the a35r6 is "Room 6".
Understand "Area 35 in Room 6" as a35r6.
the parent of the a35r6 is r6.
the X of the a35r6 is "545125.2443992454".
the Y of the a35r6 is "1648059.5559191136".

[create a44r6]
a44r6 is a area. "An area (44) in r6".
the printed name of the a44r6 is "Room 6".
Understand "Area 44 in Room 6" as a44r6.
the parent of the a44r6 is r6.
the X of the a44r6 is "545077.0".
the Y of the a44r6 is "1648114.0".

[create a45r6]
a45r6 is a area. "An area (45) in r6".
the printed name of the a45r6 is "Room 6".
Understand "Area 45 in Room 6" as a45r6.
the parent of the a45r6 is r6.
the X of the a45r6 is "545107.9153780066".
the Y of the a45r6 is "1648088.1834621995".

[create a46r6]
a46r6 is a area. "An area (46) in r6".
the printed name of the a46r6 is "Room 6".
Understand "Area 46 in Room 6" as a46r6.
the parent of the a46r6 is r6.
the X of the a46r6 is "545888.0".
the Y of the a46r6 is "1648114.0".

[create a47r6]
a47r6 is a area. "An area (47) in r6".
the printed name of the a47r6 is "Room 6".
Understand "Area 47 in Room 6" as a47r6.
the parent of the a47r6 is r6.
the X of the a47r6 is "545856.9123931625".
the Y of the a47r6 is "1648088.1662393163".

[create a56r6]
a56r6 is a area. "An area (56) in r6".
the printed name of the a56r6 is "Room 6".
Understand "Area 56 in Room 6" as a56r6.
the parent of the a56r6 is r6.
the X of the a56r6 is "545897.0".
the Y of the a56r6 is "1648024.0".

[create a57r6]
a57r6 is a area. "An area (57) in r6".
the printed name of the a57r6 is "Room 6".
Understand "Area 57 in Room 6" as a57r6.
the parent of the a57r6 is r6.
the X of the a57r6 is "545854.94".
the Y of the a57r6 is "1648042.519".

[create a86r6]
a86r6 is a area. "An area (86) in r6".
the printed name of the a86r6 is "Room 6".
Understand "Area 86 in Room 6" as a86r6.
the parent of the a86r6 is r6.
the X of the a86r6 is "545068.0".
the Y of the a86r6 is "1648024.0".

[create a87r6]
a87r6 is a area. "An area (87) in r6".
the printed name of the a87r6 is "Room 6".
Understand "Area 87 in Room 6" as a87r6.
the parent of the a87r6 is r6.
the X of the a87r6 is "545110.6911764706".
the Y of the a87r6 is "1648042.4558823528".

[create d0]
d0 is a door. "A door between a2r0 and a3r1".
d0 is southeast of a2r0 and northwest of a3r1.

[create d2]
d2 is a door. "A door between a3r1 and a4r2".
d2 is north of a3r1 and south of a4r2.

[create d4]
d4 is a door. "A door between a8r5 and a7r6".
d4 is north of a8r5 and south of a7r6.

[create d6]
d6 is a door. "A door between a41r0 and a39r5".
d6 is south of a41r0 and north of a39r5.

[create d7]
d7 is a door. "A door between a41r0 and a40r5".
d7 is southeast of a41r0 and northwest of a40r5.

[create d10]
d10 is a door. "A door between a0r3 and a24r4".
d10 is northeast of a0r3 and southwest of a24r4.

[create d12]
d12 is a door. "A door between a4r2 and a0r3".
d12 is southeast of a4r2 and northwest of a0r3.

south of a0r3 is north of a1r3.

north of a5r5 is south of a6r5.

north of a6r5 is south of a9r5.

south of a10r5 is north of a11r5.

north of a9r5 is south of a12r5.

east of a13r5 is west of a14r5.

east of a8r5 is west of a15r5.

southeast of a16r1 is northwest of a17r1.

northeast of a15r5 is southwest of a18r5.

northeast of a18r5 is southwest of a19r5.

south of a5r5 is north of a14r5.

southwest of a16r1 is northeast of a20r1.

south of a3r1 is north of a16r1.

east of a14r5 is west of a21r5.

east of a1r3 is west of a22r3.

west of a1r3 is east of a23r3.

east of a24r4 is west of a25r4.

east of a26r2 is west of a27r2.

east of a28r2 is west of a29r2.

southwest of a28r2 is northeast of a30r2.

east of a31r0 is west of a32r0.

southeast of a32r0 is northwest of a33r0.

southeast of a27r2 is northwest of a30r2.

east of a7r6 is west of a34r6.

west of a7r6 is east of a35r6.

northeast of a36r0 is southwest of a37r0.

west of a24r4 is east of a38r4.

north of a10r5 is south of a39r5.

north of a12r5 is south of a40r5.

south of a33r0 is north of a41r0.

north of a4r2 is south of a30r2.

southwest of a42r4 is northeast of a43r4.

southeast of a44r6 is northwest of a45r6.

southwest of a46r6 is northeast of a47r6.

southeast of a48r0 is northwest of a49r0.

east of a50r0 is west of a51r0.

west of a52r2 is east of a53r2.

east of a54r2 is west of a55r2.

northwest of a56r6 is southeast of a57r6.

northeast of a58r4 is southwest of a59r4.

northwest of a60r4 is southeast of a61r4.

northeast of a62r0 is southwest of a63r0.

southeast of a64r2 is northwest of a65r2.

southwest of a66r2 is northeast of a67r2.

southwest of a68r0 is northeast of a69r0.

southeast of a70r4 is northwest of a71r4.

southeast of a72r3 is northwest of a73r3.

northwest of a74r3 is southeast of a75r3.

southwest of a76r3 is northeast of a77r3.

northwest of a78r5 is southeast of a79r5.

northeast of a80r5 is southwest of a81r5.

northeast of a82r2 is southwest of a83r2.

northwest of a84r2 is southeast of a85r2.

northeast of a86r6 is southwest of a87r6.

northwest of a88r5 is southeast of a89r5.

northeast of a90r3 is southwest of a91r3.

northeast of a92r5 is southwest of a93r5.

southwest of a94r5 is northeast of a95r5.

southeast of a17r1 is northwest of a96r1.

south of a20r1 is north of a97r1.

northwest of a38r4 is southeast of a71r4.

northeast of a25r4 is southwest of a43r4.

northwest of a35r6 is southeast of a45r6.

northeast of a34r6 is southwest of a47r6.

northwest of a31r0 is southeast of a49r0.

north of a32r0 is south of a51r0.

north of a28r2 is south of a53r2.

north of a27r2 is south of a55r2.

southwest of a35r6 is northeast of a87r6.

southeast of a34r6 is northwest of a57r6.

southwest of a38r4 is northeast of a59r4.

southeast of a25r4 is northwest of a61r4.

southwest of a31r0 is northeast of a63r0.

northwest of a26r2 is southeast of a65r2.

northeast of a29r2 is southwest of a67r2.

northeast of a37r0 is southwest of a69r0.

southeast of a15r5 is northwest of a79r5.

northwest of a13r5 is southeast of a18r5.

southwest of a8r5 is northeast of a81r5.

south of a11r5 is north of a19r5.

west of a2r0 is east of a36r0.

east of a33r0 is west of a36r0.

southwest of a26r2 is northeast of a83r2.

southeast of a29r2 is northwest of a85r2.

southeast of a21r5 is northwest of a89r5.

southwest of a23r3 is northeast of a91r3.

southwest of a13r5 is northeast of a93r5.

northeast of a21r5 is southwest of a95r5.

northwest of a23r3 is southeast of a73r3.

southeast of a22r3 is northwest of a75r3.

northeast of a22r3 is southwest of a77r3.

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
the player is in a4r2.

the orientation of the player is 6.
