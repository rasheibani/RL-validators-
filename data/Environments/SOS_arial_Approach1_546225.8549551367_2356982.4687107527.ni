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
the X of the a0r0 is "547544.1983938243".
the Y of the a0r0 is "2357177.4374849973".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "547617.8953955586".
the Y of the a1r0 is "2357102.831462888".

[create a2r0]
a2r0 is a area. "An area (2) in r0".
the printed name of the a2r0 is "Room 0".
Understand "Area 2 in Room 0" as a2r0.
the parent of the a2r0 is r0.
the X of the a2r0 is "545286.8509703097".
the Y of the a2r0 is "2357210.9535603453".

[create a3r0]
a3r0 is a area. "An area (3) in r0".
the printed name of the a3r0 is "Room 0".
Understand "Area 3 in Room 0" as a3r0.
the parent of the a3r0 is r0.
the X of the a3r0 is "545381.1400600461".
the Y of the a3r0 is "2357336.181603638".

[create a4r0]
a4r0 is a area. "An area (4) in r0".
the printed name of the a4r0 is "Room 0".
Understand "Area 4 in Room 0" as a4r0.
the parent of the a4r0 is r0.
the X of the a4r0 is "545175.7493486579".
the Y of the a4r0 is "2357308.335588297".

[create a5r0]
a5r0 is a area. "An area (5) in r0".
the printed name of the a5r0 is "Room 0".
Understand "Area 5 in Room 0" as a5r0.
the parent of the a5r0 is r0.
the X of the a5r0 is "547416.7493486579".
the Y of the a5r0 is "2357308.335588297".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "547535.3942510465".
the Y of the a6r0 is "2357204.6067453553".

[create a7r0]
a7r0 is a area. "An area (7) in r0".
the printed name of the a7r0 is "Room 0".
Understand "Area 7 in Room 0" as a7r0.
the parent of the a7r0 is r0.
the X of the a7r0 is "547601.663296771".
the Y of the a7r0 is "2357301.3419554993".

[create a8r0]
a8r0 is a area. "An area (8) in r0".
the printed name of the a8r0 is "Room 0".
Understand "Area 8 in Room 0" as a8r0.
the parent of the a8r0 is r0.
the X of the a8r0 is "546606.1358759394".
the Y of the a8r0 is "2357571.604479606".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "546744.0262927371".
the Y of the a9r0 is "2357378.0818879185".

[create a10r0]
a10r0 is a area. "An area (10) in r0".
the printed name of the a10r0 is "Room 0".
Understand "Area 10 in Room 0" as a10r0.
the parent of the a10r0 is r0.
the X of the a10r0 is "548847.1358759396".
the Y of the a10r0 is "2357571.604479606".

[create a11r0]
a11r0 is a area. "An area (11) in r0".
the printed name of the a11r0 is "Room 0".
Understand "Area 11 in Room 0" as a11r0.
the parent of the a11r0 is r0.
the X of the a11r0 is "548985.026292737".
the Y of the a11r0 is "2357378.081887919".

[create a12r0]
a12r0 is a area. "An area (12) in r0".
the printed name of the a12r0 is "Room 0".
Understand "Area 12 in Room 0" as a12r0.
the parent of the a12r0 is r0.
the X of the a12r0 is "546025.530407923".
the Y of the a12r0 is "2357438.3784869965".

[create a14r0]
a14r0 is a area. "An area (14) in r0".
the printed name of the a14r0 is "Room 0".
Understand "Area 14 in Room 0" as a14r0.
the parent of the a14r0 is r0.
the X of the a14r0 is "546248.7193494682".
the Y of the a14r0 is "2357003.8378809607".

[create a15r0]
a15r0 is a area. "An area (15) in r0".
the printed name of the a15r0 is "Room 0".
Understand "Area 15 in Room 0" as a15r0.
the parent of the a15r0 is r0.
the X of the a15r0 is "546375.7806968183".
the Y of the a15r0 is "2356829.2959083957".

[create a16r0]
a16r0 is a area. "An area (16) in r0".
the printed name of the a16r0 is "Room 0".
Understand "Area 16 in Room 0" as a16r0.
the parent of the a16r0 is r0.
the X of the a16r0 is "546578.6026086037".
the Y of the a16r0 is "2356821.7193912915".

[create a17r0]
a17r0 is a area. "An area (17) in r0".
the printed name of the a17r0 is "Room 0".
Understand "Area 17 in Room 0" as a17r0.
the parent of the a17r0 is r0.
the X of the a17r0 is "546748.8915797039".
the Y of the a17r0 is "2357115.6371144797".

[create a18r0]
a18r0 is a area. "An area (18) in r0".
the printed name of the a18r0 is "Room 0".
Understand "Area 18 in Room 0" as a18r0.
the parent of the a18r0 is r0.
the X of the a18r0 is "547610.5662136555".
the Y of the a18r0 is "2356988.698007508".

[create a19r0]
a19r0 is a area. "An area (19) in r0".
the printed name of the a19r0 is "Room 0".
Understand "Area 19 in Room 0" as a19r0.
the parent of the a19r0 is r0.
the X of the a19r0 is "547714.2560604885".
the Y of the a19r0 is "2356885.681275785".

[create a20r0]
a20r0 is a area. "An area (20) in r0".
the printed name of the a20r0 is "Room 0".
Understand "Area 20 in Room 0" as a20r0.
the parent of the a20r0 is r0.
the X of the a20r0 is "545287.3706247588".
the Y of the a20r0 is "2357169.8536175424".

[create a21r0]
a21r0 is a area. "An area (21) in r0".
the printed name of the a21r0 is "Room 0".
Understand "Area 21 in Room 0" as a21r0.
the parent of the a21r0 is r0.
the X of the a21r0 is "545470.1444115917".
the Y of the a21r0 is "2356887.3950302876".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "548989.9429831192".
the Y of the a22r0 is "2357113.9704575916".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "547807.7576253854".
the Y of the a23r0 is "2357340.4166666665".

[create a24r0]
a24r0 is a area. "An area (24) in r0".
the printed name of the a24r0 is "Room 0".
Understand "Area 24 in Room 0" as a24r0.
the parent of the a24r0 is r0.
the X of the a24r0 is "548839.2118265713".
the Y of the a24r0 is "2356876.839089826".

[create a49r0]
a49r0 is a area. "An area (49) in r0".
the printed name of the a49r0 is "Room 0".
Understand "Area 49 in Room 0" as a49r0.
the parent of the a49r0 is r0.
the X of the a49r0 is "548929.0".
the Y of the a49r0 is "2356770.0".

[create a50r0]
a50r0 is a area. "An area (50) in r0".
the printed name of the a50r0 is "Room 0".
Understand "Area 50 in Room 0" as a50r0.
the parent of the a50r0 is r0.
the X of the a50r0 is "548847.1435490518".
the Y of the a50r0 is "2356867.690972592".

[create a51r0]
a51r0 is a area. "An area (51) in r0".
the printed name of the a51r0 is "Room 0".
Understand "Area 51 in Room 0" as a51r0.
the parent of the a51r0 is r0.
the X of the a51r0 is "547594.7404755817".
the Y of the a51r0 is "2356804.829711071".

[create a52r0]
a52r0 is a area. "An area (52) in r0".
the printed name of the a52r0 is "Room 0".
Understand "Area 52 in Room 0" as a52r0.
the parent of the a52r0 is r0.
the X of the a52r0 is "547683.9313763478".
the Y of the a52r0 is "2356841.3693180163".

[create a53r0]
a53r0 is a area. "An area (53) in r0".
the printed name of the a53r0 is "Room 0".
Understand "Area 53 in Room 0" as a53r0.
the parent of the a53r0 is r0.
the X of the a53r0 is "545374.0".
the Y of the a53r0 is "2356774.0".

[create a54r0]
a54r0 is a area. "An area (54) in r0".
the printed name of the a54r0 is "Room 0".
Understand "Area 54 in Room 0" as a54r0.
the parent of the a54r0 is r0.
the X of the a54r0 is "545454.8005460426".
the Y of the a54r0 is "2356868.859406447".

[create a55r0]
a55r0 is a area. "An area (55) in r0".
the printed name of the a55r0 is "Room 0".
Understand "Area 55 in Room 0" as a55r0.
the parent of the a55r0 is r0.
the X of the a55r0 is "546129.0".
the Y of the a55r0 is "2356895.0".

[create a56r0]
a56r0 is a area. "An area (56) in r0".
the printed name of the a56r0 is "Room 0".
Understand "Area 56 in Room 0" as a56r0.
the parent of the a56r0 is r0.
the X of the a56r0 is "546225.8549551367".
the Y of the a56r0 is "2356982.4687107527".

[create a57r0]
a57r0 is a area. "An area (57) in r0".
the printed name of the a57r0 is "Room 0".
Understand "Area 57 in Room 0" as a57r0.
the parent of the a57r0 is r0.
the X of the a57r0 is "549130.0".
the Y of the a57r0 is "2357075.0".

[create a58r0]
a58r0 is a area. "An area (58) in r0".
the printed name of the a58r0 is "Room 0".
Understand "Area 58 in Room 0" as a58r0.
the parent of the a58r0 is r0.
the X of the a58r0 is "549019.8668325052".
the Y of the a58r0 is "2357103.833333333".

[create a59r0]
a59r0 is a area. "An area (59) in r0".
the printed name of the a59r0 is "Room 0".
Understand "Area 59 in Room 0" as a59r0.
the parent of the a59r0 is r0.
the X of the a59r0 is "546889.0".
the Y of the a59r0 is "2357075.0".

[create a60r0]
a60r0 is a area. "An area (60) in r0".
the printed name of the a60r0 is "Room 0".
Understand "Area 60 in Room 0" as a60r0.
the parent of the a60r0 is r0.
the X of the a60r0 is "546785.6681741565".
the Y of the a60r0 is "2357103.833333333".

[create a75r0]
a75r0 is a area. "An area (75) in r0".
the printed name of the a75r0 is "Room 0".
Understand "Area 75 in Room 0" as a75r0.
the parent of the a75r0 is r0.
the X of the a75r0 is "545877.0".
the Y of the a75r0 is "2357406.0".

[create a76r0]
a76r0 is a area. "An area (76) in r0".
the printed name of the a76r0 is "Room 0".
Understand "Area 76 in Room 0" as a76r0.
the parent of the a76r0 is r0.
the X of the a76r0 is "546020.7497077838".
the Y of the a76r0 is "2357437.1856435644".

[create a77r0]
a77r0 is a area. "An area (77) in r0".
the printed name of the a77r0 is "Room 0".
Understand "Area 77 in Room 0" as a77r0.
the parent of the a77r0 is r0.
the X of the a77r0 is "546706.0".
the Y of the a77r0 is "2357689.0".

[create a78r0]
a78r0 is a area. "An area (78) in r0".
the printed name of the a78r0 is "Room 0".
Understand "Area 78 in Room 0" as a78r0.
the parent of the a78r0 is r0.
the X of the a78r0 is "546624.2628214923".
the Y of the a78r0 is "2357593.735192259".

[create a79r0]
a79r0 is a area. "An area (79) in r0".
the printed name of the a79r0 is "Room 0".
Understand "Area 79 in Room 0" as a79r0.
the parent of the a79r0 is r0.
the X of the a79r0 is "548947.0".
the Y of the a79r0 is "2357689.0".

[create a80r0]
a80r0 is a area. "An area (80) in r0".
the printed name of the a80r0 is "Room 0".
Understand "Area 80 in Room 0" as a80r0.
the parent of the a80r0 is r0.
the X of the a80r0 is "548865.2628214925".
the Y of the a80r0 is "2357593.7351922593".

[create a91r0]
a91r0 is a area. "An area (91) in r0".
the printed name of the a91r0 is "Room 0".
Understand "Area 91 in Room 0" as a91r0.
the parent of the a91r0 is r0.
the X of the a91r0 is "545144.0".
the Y of the a91r0 is "2357124.0".

[create a92r0]
a92r0 is a area. "An area (92) in r0".
the printed name of the a92r0 is "Room 0".
Understand "Area 92 in Room 0" as a92r0.
the parent of the a92r0 is r0.
the X of the a92r0 is "545268.5587628874".
the Y of the a92r0 is "2357164.090996564".

[create a93r0]
a93r0 is a area. "An area (93) in r0".
the printed name of the a93r0 is "Room 0".
Understand "Area 93 in Room 0" as a93r0.
the parent of the a93r0 is r0.
the X of the a93r0 is "547385.0".
the Y of the a93r0 is "2357124.0".

[create a94r0]
a94r0 is a area. "An area (94) in r0".
the printed name of the a94r0 is "Room 0".
Understand "Area 94 in Room 0" as a94r0.
the parent of the a94r0 is r0.
the X of the a94r0 is "547486.9410344702".
the Y of the a94r0 is "2357163.4741494246".

[create a95r0]
a95r0 is a area. "An area (95) in r0".
the printed name of the a95r0 is "Room 0".
Understand "Area 95 in Room 0" as a95r0.
the parent of the a95r0 is r0.
the X of the a95r0 is "547663.0908519333".
the Y of the a95r0 is "2357288.096196165".

[create a96r0]
a96r0 is a area. "An area (96) in r0".
the printed name of the a96r0 is "Room 0".
Understand "Area 96 in Room 0" as a96r0.
the parent of the a96r0 is r0.
the X of the a96r0 is "547659.9745288967".
the Y of the a96r0 is "2357307.9761675373".

[create a97r0]
a97r0 is a area. "An area (97) in r0".
the printed name of the a97r0 is "Room 0".
Understand "Area 97 in Room 0" as a97r0.
the parent of the a97r0 is r0.
the X of the a97r0 is "547654.0".
the Y of the a97r0 is "2357368.0".

[create a98r0]
a98r0 is a area. "An area (98) in r0".
the printed name of the a98r0 is "Room 0".
Understand "Area 98 in Room 0" as a98r0.
the parent of the a98r0 is r0.
the X of the a98r0 is "547621.4522777337".
the Y of the a98r0 is "2357344.0624426357".

[create a99r0]
a99r0 is a area. "An area (99) in r0".
the printed name of the a99r0 is "Room 0".
Understand "Area 99 in Room 0" as a99r0.
the parent of the a99r0 is r0.
the X of the a99r0 is "547680.0".
the Y of the a99r0 is "2357306.0".

[create a100r0]
a100r0 is a area. "An area (100) in r0".
the printed name of the a100r0 is "Room 0".
Understand "Area 100 in Room 0" as a100r0.
the parent of the a100r0 is r0.
the X of the a100r0 is "547693.3124691218".
the Y of the a100r0 is "2357340.416666667".

[create a101r0]
a101r0 is a area. "An area (101) in r0".
the printed name of the a101r0 is "Room 0".
Understand "Area 101 in Room 0" as a101r0.
the parent of the a101r0 is r0.
the X of the a101r0 is "546889.0".
the Y of the a101r0 is "2357422.0".

[create a102r0]
a102r0 is a area. "An area (102) in r0".
the printed name of the a102r0 is "Room 0".
Understand "Area 102 in Room 0" as a102r0.
the parent of the a102r0 is r0.
the X of the a102r0 is "546789.4327868854".
the Y of the a102r0 is "2357393.0".

[create a103r0]
a103r0 is a area. "An area (103) in r0".
the printed name of the a103r0 is "Room 0".
Understand "Area 103 in Room 0" as a103r0.
the parent of the a103r0 is r0.
the X of the a103r0 is "549130.0".
the Y of the a103r0 is "2357422.0".

[create a104r0]
a104r0 is a area. "An area (104) in r0".
the printed name of the a104r0 is "Room 0".
Understand "Area 104 in Room 0" as a104r0.
the parent of the a104r0 is r0.
the X of the a104r0 is "549030.4327868852".
the Y of the a104r0 is "2357393.0".

[create a105r0]
a105r0 is a area. "An area (105) in r0".
the printed name of the a105r0 is "Room 0".
Understand "Area 105 in Room 0" as a105r0.
the parent of the a105r0 is r0.
the X of the a105r0 is "547379.0".
the Y of the a105r0 is "2357344.0".

[create a111r0]
a111r0 is a area. "An area (111) in r0".
the printed name of the a111r0 is "Room 0".
Understand "Area 111 in Room 0" as a111r0.
the parent of the a111r0 is r0.
the X of the a111r0 is "545413.0".
the Y of the a111r0 is "2357368.0".

[create a112r0]
a112r0 is a area. "An area (112) in r0".
the printed name of the a112r0 is "Room 0".
Understand "Area 112 in Room 0" as a112r0.
the parent of the a112r0 is r0.
the X of the a112r0 is "545138.0".
the Y of the a112r0 is "2357344.0".

[create a13r1]
a13r1 is a area. "An area (13) in r1".
the printed name of the a13r1 is "Room 1".
Understand "Area 13 in Room 1" as a13r1.
the parent of the a13r1 is r1.
the X of the a13r1 is "546053.7143569186".
the Y of the a13r1 is "2357732.2645028583".

[create a25r1]
a25r1 is a area. "An area (25) in r1".
the printed name of the a25r1 is "Room 1".
Understand "Area 25 in Room 1" as a25r1.
the parent of the a25r1 is r1.
the X of the a25r1 is "546537.4078773585".
the Y of the a25r1 is "2358667.5110657266".

[create a26r1]
a26r1 is a area. "An area (26) in r1".
the printed name of the a26r1 is "Room 1".
Understand "Area 26 in Room 1" as a26r1.
the parent of the a26r1 is r1.
the X of the a26r1 is "547619.5634312476".
the Y of the a26r1 is "2358454.0817658743".

[create a27r1]
a27r1 is a area. "An area (27) in r1".
the printed name of the a27r1 is "Room 1".
Understand "Area 27 in Room 1" as a27r1.
the parent of the a27r1 is r1.
the X of the a27r1 is "547749.87502293".
the Y of the a27r1 is "2358580.778454636".

[create a28r1]
a28r1 is a area. "An area (28) in r1".
the printed name of the a28r1 is "Room 1".
Understand "Area 28 in Room 1" as a28r1.
the parent of the a28r1 is r1.
the X of the a28r1 is "548761.4939048442".
the Y of the a28r1 is "2358572.2869766476".

[create a29r1]
a29r1 is a area. "An area (29) in r1".
the printed name of the a29r1 is "Room 1".
Understand "Area 29 in Room 1" as a29r1.
the parent of the a29r1 is r1.
the X of the a29r1 is "545507.4245925718".
the Y of the a29r1 is "2358577.5573323336".

[create a30r1]
a30r1 is a area. "An area (30) in r1".
the printed name of the a30r1 is "Room 1".
Understand "Area 30 in Room 1" as a30r1.
the parent of the a30r1 is r1.
the X of the a30r1 is "546394.5482646527".
the Y of the a30r1 is "2358598.6613371726".

[create a31r1]
a31r1 is a area. "An area (31) in r1".
the printed name of the a31r1 is "Room 1".
Understand "Area 31 in Room 1" as a31r1.
the parent of the a31r1 is r1.
the X of the a31r1 is "546503.75".
the Y of the a31r1 is "2358771.4896360226".

[create a32r1]
a32r1 is a area. "An area (32) in r1".
the printed name of the a32r1 is "Room 1".
Understand "Area 32 in Room 1" as a32r1.
the parent of the a32r1 is r1.
the X of the a32r1 is "547680.0507499175".
the Y of the a32r1 is "2358157.0103812153".

[create a33r1]
a33r1 is a area. "An area (33) in r1".
the printed name of the a33r1 is "Room 1".
Understand "Area 33 in Room 1" as a33r1.
the parent of the a33r1 is r1.
the X of the a33r1 is "548913.2399632924".
the Y of the a33r1 is "2358355.700185554".

[create a34r1]
a34r1 is a area. "An area (34) in r1".
the printed name of the a34r1 is "Room 1".
Understand "Area 34 in Room 1" as a34r1.
the parent of the a34r1 is r1.
the X of the a34r1 is "549026.8841058412".
the Y of the a34r1 is "2358248.760473517".

[create a35r1]
a35r1 is a area. "An area (35) in r1".
the printed name of the a35r1 is "Room 1".
Understand "Area 35 in Room 1" as a35r1.
the parent of the a35r1 is r1.
the X of the a35r1 is "546672.0558428635".
the Y of the a35r1 is "2358355.8882781235".

[create a36r1]
a36r1 is a area. "An area (36) in r1".
the printed name of the a36r1 is "Room 1".
Understand "Area 36 in Room 1" as a36r1.
the parent of the a36r1 is r1.
the X of the a36r1 is "546785.8841058412".
the Y of the a36r1 is "2358248.760473517".

[create a37r1]
a37r1 is a area. "An area (37) in r1".
the printed name of the a37r1 is "Room 1".
Understand "Area 37 in Room 1" as a37r1.
the parent of the a37r1 is r1.
the X of the a37r1 is "546576.4691805373".
the Y of the a37r1 is "2358236.061181504".

[create a38r1]
a38r1 is a area. "An area (38) in r1".
the printed name of the a38r1 is "Room 1".
Understand "Area 38 in Room 1" as a38r1.
the parent of the a38r1 is r1.
the X of the a38r1 is "548817.4691805374".
the Y of the a38r1 is "2358236.0611815033".

[create a39r1]
a39r1 is a area. "An area (39) in r1".
the printed name of the a39r1 is "Room 1".
Understand "Area 39 in Room 1" as a39r1.
the parent of the a39r1 is r1.
the X of the a39r1 is "548912.9844760683".
the Y of the a39r1 is "2358355.793564191".

[create a40r1]
a40r1 is a area. "An area (40) in r1".
the printed name of the a40r1 is "Room 1".
Understand "Area 40 in Room 1" as a40r1.
the parent of the a40r1 is r1.
the X of the a40r1 is "546503.8581175047".
the Y of the a40r1 is "2358617.432159452".

[create a41r1]
a41r1 is a area. "An area (41) in r1".
the printed name of the a41r1 is "Room 1".
Understand "Area 41 in Room 1" as a41r1.
the parent of the a41r1 is r1.
the X of the a41r1 is "546672.0275332965".
the Y of the a41r1 is "2358356.446105271".

[create a42r1]
a42r1 is a area. "An area (42) in r1".
the printed name of the a42r1 is "Room 1".
Understand "Area 42 in Room 1" as a42r1.
the parent of the a42r1 is r1.
the X of the a42r1 is "545363.4620661372".
the Y of the a42r1 is "2358366.678492637".

[create a43r1]
a43r1 is a area. "An area (43) in r1".
the printed name of the a43r1 is "Room 1".
Understand "Area 43 in Room 1" as a43r1.
the parent of the a43r1 is r1.
the X of the a43r1 is "547864.5085053628".
the Y of the a43r1 is "2357832.7308673826".

[create a44r1]
a44r1 is a area. "An area (44) in r1".
the printed name of the a44r1 is "Room 1".
Understand "Area 44 in Room 1" as a44r1.
the parent of the a44r1 is r1.
the X of the a44r1 is "545477.7356304465".
the Y of the a44r1 is "2357966.6963967634".

[create a45r1]
a45r1 is a area. "An area (45) in r1".
the printed name of the a45r1 is "Room 1".
Understand "Area 45 in Room 1" as a45r1.
the parent of the a45r1 is r1.
the X of the a45r1 is "546002.1402841739".
the Y of the a45r1 is "2357801.0770813".

[create a46r1]
a46r1 is a area. "An area (46) in r1".
the printed name of the a46r1 is "Room 1".
Understand "Area 46 in Room 1" as a46r1.
the parent of the a46r1 is r1.
the X of the a46r1 is "546026.8259492011".
the Y of the a46r1 is "2358169.4964010664".

[create a47r1]
a47r1 is a area. "An area (47) in r1".
the printed name of the a47r1 is "Room 1".
Understand "Area 47 in Room 1" as a47r1.
the parent of the a47r1 is r1.
the X of the a47r1 is "545360.6355807784".
the Y of the a47r1 is "2358134.54392795".

[create a48r1]
a48r1 is a area. "An area (48) in r1".
the printed name of the a48r1 is "Room 1".
Understand "Area 48 in Room 1" as a48r1.
the parent of the a48r1 is r1.
the X of the a48r1 is "547762.3170050862".
the Y of the a48r1 is "2358011.069752747".

[create a61r1]
a61r1 is a area. "An area (61) in r1".
the printed name of the a61r1 is "Room 1".
Understand "Area 61 in Room 1" as a61r1.
the parent of the a61r1 is r1.
the X of the a61r1 is "547463.0".
the Y of the a61r1 is "2358093.0".

[create a62r1]
a62r1 is a area. "An area (62) in r1".
the printed name of the a62r1 is "Room 1".
Understand "Area 62 in Room 1" as a62r1.
the parent of the a62r1 is r1.
the X of the a62r1 is "547587.520967742".
the Y of the a62r1 is "2358130.5".

[create a63r1]
a63r1 is a area. "An area (63) in r1".
the printed name of the a63r1 is "Room 1".
Understand "Area 63 in Room 1" as a63r1.
the parent of the a63r1 is r1.
the X of the a63r1 is "549056.0".
the Y of the a63r1 is "2358402.0".

[create a64r1]
a64r1 is a area. "An area (64) in r1".
the printed name of the a64r1 is "Room 1".
Understand "Area 64 in Room 1" as a64r1.
the parent of the a64r1 is r1.
the X of the a64r1 is "548943.541997861".
the Y of the a64r1 is "2358366.664849723".

[create a65r1]
a65r1 is a area. "An area (65) in r1".
the printed name of the a65r1 is "Room 1".
Understand "Area 65 in Room 1" as a65r1.
the parent of the a65r1 is r1.
the X of the a65r1 is "546815.0".
the Y of the a65r1 is "2358402.0".

[create a66r1]
a66r1 is a area. "An area (66) in r1".
the printed name of the a66r1 is "Room 1".
Understand "Area 66 in Room 1" as a66r1.
the parent of the a66r1 is r1.
the X of the a66r1 is "546699.2779609922".
the Y of the a66r1 is "2358366.521375575".

[create a67r1]
a67r1 is a area. "An area (67) in r1".
the printed name of the a67r1 is "Room 1".
Understand "Area 67 in Room 1" as a67r1.
the parent of the a67r1 is r1.
the X of the a67r1 is "545222.0".
the Y of the a67r1 is "2358407.0".

[create a68r1]
a68r1 is a area. "An area (68) in r1".
the printed name of the a68r1 is "Room 1".
Understand "Area 68 in Room 1" as a68r1.
the parent of the a68r1 is r1.
the X of the a68r1 is "545325.093121693".
the Y of the a68r1 is "2358379.6666666665".

[create a69r1]
a69r1 is a area. "An area (69) in r1".
the printed name of the a69r1 is "Room 1".
Understand "Area 69 in Room 1" as a69r1.
the parent of the a69r1 is r1.
the X of the a69r1 is "548855.0".
the Y of the a69r1 is "2358691.0".

[create a70r1]
a70r1 is a area. "An area (70) in r1".
the printed name of the a70r1 is "Room 1".
Understand "Area 70 in Room 1" as a70r1.
the parent of the a70r1 is r1.
the X of the a70r1 is "548781.37187896".
the Y of the a70r1 is "2358596.911929657".

[create a71r1]
a71r1 is a area. "An area (71) in r1".
the printed name of the a71r1 is "Room 1".
Understand "Area 71 in Room 1" as a71r1.
the parent of the a71r1 is r1.
the X of the a71r1 is "545411.0".
the Y of the a71r1 is "2358692.0".

[create a72r1]
a72r1 is a area. "An area (72) in r1".
the printed name of the a72r1 is "Room 1".
Understand "Area 72 in Room 1" as a72r1.
the parent of the a72r1 is r1.
the X of the a72r1 is "545491.9517547006".
the Y of the a72r1 is "2358597.2825205666".

[create a73r1]
a73r1 is a area. "An area (73) in r1".
the printed name of the a73r1 is "Room 1".
Understand "Area 73 in Room 1" as a73r1.
the parent of the a73r1 is r1.
the X of the a73r1 is "547652.0".
the Y of the a73r1 is "2358692.0".

[create a74r1]
a74r1 is a area. "An area (74) in r1".
the printed name of the a74r1 is "Room 1".
Understand "Area 74 in Room 1" as a74r1.
the parent of the a74r1 is r1.
the X of the a74r1 is "547724.0037760558".
the Y of the a74r1 is "2358619.229251971".

[create a81r1]
a81r1 is a area. "An area (81) in r1".
the printed name of the a81r1 is "Room 1".
Understand "Area 81 in Room 1" as a81r1.
the parent of the a81r1 is r1.
the X of the a81r1 is "545377.0".
the Y of the a81r1 is "2357852.0".

[create a82r1]
a82r1 is a area. "An area (82) in r1".
the printed name of the a82r1 is "Room 1".
Understand "Area 82 in Room 1" as a82r1.
the parent of the a82r1 is r1.
the X of the a82r1 is "545459.4667714846".
the Y of the a82r1 is "2357947.624894523".

[create a83r1]
a83r1 is a area. "An area (83) in r1".
the printed name of the a83r1 is "Room 1".
Understand "Area 83 in Room 1" as a83r1.
the parent of the a83r1 is r1.
the X of the a83r1 is "547980.0".
the Y of the a83r1 is "2358041.0".

[create a84r1]
a84r1 is a area. "An area (84) in r1".
the printed name of the a84r1 is "Room 1".
Understand "Area 84 in Room 1" as a84r1.
the parent of the a84r1 is r1.
the X of the a84r1 is "547835.8957332749".
the Y of the a84r1 is "2358009.092253521".

[create a85r1]
a85r1 is a area. "An area (85) in r1".
the printed name of the a85r1 is "Room 1".
Understand "Area 85 in Room 1" as a85r1.
the parent of the a85r1 is r1.
the X of the a85r1 is "547680.0".
the Y of the a85r1 is "2357820.0127388537".

[create a86r1]
a86r1 is a area. "An area (86) in r1".
the printed name of the a86r1 is "Room 1".
Understand "Area 86 in Room 1" as a86r1.
the parent of the a86r1 is r1.
the X of the a86r1 is "547709.6257381698".
the Y of the a86r1 is "2357953.5155162504".

[create a87r1]
a87r1 is a area. "An area (87) in r1".
the printed name of the a87r1 is "Room 1".
Understand "Area 87 in Room 1" as a87r1.
the parent of the a87r1 is r1.
the X of the a87r1 is "545222.0".
the Y of the a87r1 is "2358093.0".

[create a88r1]
a88r1 is a area. "An area (88) in r1".
the printed name of the a88r1 is "Room 1".
Understand "Area 88 in Room 1" as a88r1.
the parent of the a88r1 is r1.
the X of the a88r1 is "545346.5209677418".
the Y of the a88r1 is "2358130.5".

[create a89r1]
a89r1 is a area. "An area (89) in r1".
the printed name of the a89r1 is "Room 1".
Understand "Area 89 in Room 1" as a89r1.
the parent of the a89r1 is r1.
the X of the a89r1 is "545877.0".
the Y of the a89r1 is "2358230.0".

[create a90r1]
a90r1 is a area. "An area (90) in r1".
the printed name of the a90r1 is "Room 1".
Understand "Area 90 in Room 1" as a90r1.
the parent of the a90r1 is r1.
the X of the a90r1 is "545964.3470245568".
the Y of the a90r1 is "2358194.5836267606".

[create a106r1]
a106r1 is a area. "An area (106) in r1".
the printed name of the a106r1 is "Room 1".
Understand "Area 106 in Room 1" as a106r1.
the parent of the a106r1 is r1.
the X of the a106r1 is "549064.0".
the Y of the a106r1 is "2358220.0".

[create a107r1]
a107r1 is a area. "An area (107) in r1".
the printed name of the a107r1 is "Room 1".
Understand "Area 107 in Room 1" as a107r1.
the parent of the a107r1 is r1.
the X of the a107r1 is "546823.0".
the Y of the a107r1 is "2358220.0".

[create a108r1]
a108r1 is a area. "An area (108) in r1".
the printed name of the a108r1 is "Room 1".
Understand "Area 108 in Room 1" as a108r1.
the parent of the a108r1 is r1.
the X of the a108r1 is "546544.0".
the Y of the a108r1 is "2358200.0".

[create a109r1]
a109r1 is a area. "An area (109) in r1".
the printed name of the a109r1 is "Room 1".
Understand "Area 109 in Room 1" as a109r1.
the parent of the a109r1 is r1.
the X of the a109r1 is "548785.0".
the Y of the a109r1 is "2358200.0".

[create a110r1]
a110r1 is a area. "An area (110) in r1".
the printed name of the a110r1 is "Room 1".
Understand "Area 110 in Room 1" as a110r1.
the parent of the a110r1 is r1.
the X of the a110r1 is "546465.0".
the Y of the a110r1 is "2358840.0".

[create a113r1]
a113r1 is a area. "An area (113) in r1".
the printed name of the a113r1 is "Room 1".
Understand "Area 113 in Room 1" as a113r1.
the parent of the a113r1 is r1.
the X of the a113r1 is "547779.5412625324".
the Y of the a113r1 is "2357994.9112204".

[create d0]
d0 is a door. "A door between a12r0 and a13r1".
d0 is north of a12r0 and south of a13r1.

[create d1]
d1 is a door. "A door between a8r0 and a13r1".
d1 is west of a8r0 and east of a13r1.

[create d2]
d2 is a door. "A door between a10r0 and a43r1".
d2 is west of a10r0 and east of a43r1.

[create d3]
d3 is a door. "A door between a23r0 and a43r1".
d3 is north of a23r0 and south of a43r1.

southeast of a0r0 is northwest of a1r0.

northeast of a2r0 is southwest of a3r0.

northwest of a2r0 is southeast of a4r0.

southeast of a5r0 is northwest of a6r0.

northeast of a6r0 is southwest of a7r0.

southeast of a8r0 is northwest of a9r0.

southeast of a10r0 is northwest of a11r0.

southeast of a14r0 is northwest of a15r0.

northeast of a16r0 is southwest of a17r0.

southeast of a18r0 is northwest of a19r0.

southeast of a20r0 is northwest of a21r0.

southeast of a12r0 is northwest of a14r0.

south of a9r0 is north of a17r0.

south of a11r0 is north of a22r0.

northeast of a1r0 is southwest of a23r0.

east of a19r0 is west of a24r0.

west of a15r0 is east of a21r0.

east of a15r0 is west of a16r0.

southwest of a22r0 is northeast of a24r0.

east of a25r1 is west of a26r1.

northeast of a26r1 is southwest of a27r1.

east of a27r1 is west of a28r1.

east of a29r1 is west of a30r1.

north of a25r1 is south of a31r1.

south of a26r1 is north of a32r1.

southeast of a33r1 is northwest of a34r1.

southeast of a35r1 is northwest of a36r1.

southwest of a35r1 is northeast of a37r1.

northeast of a38r1 is southwest of a39r1.

southeast of a28r1 is northwest of a39r1.

southeast of a40r1 is northwest of a41r1.

southwest of a29r1 is northeast of a42r1.

east of a44r1 is west of a45r1.

north of a45r1 is south of a46r1.

northwest of a44r1 is southeast of a47r1.

southeast of a32r1 is northwest of a48r1.

south of a42r1 is north of a47r1.

southwest of a30r1 is northeast of a46r1.

east of a16r0 is west of a18r0.

northwest of a49r0 is southeast of a50r0.

east of a51r0 is west of a52r0.

northeast of a53r0 is southwest of a54r0.

northeast of a55r0 is southwest of a56r0.

west of a57r0 is east of a58r0.

west of a59r0 is east of a60r0.

east of a61r1 is west of a62r1.

west of a63r1 is east of a64r1.

west of a65r1 is east of a66r1.

east of a67r1 is west of a68r1.

southwest of a69r1 is northeast of a70r1.

southeast of a71r1 is northwest of a72r1.

southeast of a73r1 is northwest of a74r1.

east of a75r0 is west of a76r0.

southwest of a77r0 is northeast of a78r0.

southwest of a79r0 is northeast of a80r0.

northeast of a81r1 is southwest of a82r1.

west of a83r1 is east of a84r1.

north of a85r1 is south of a86r1.

east of a87r1 is west of a88r1.

east of a89r1 is west of a90r1.

east of a91r0 is west of a92r0.

east of a93r0 is west of a94r0.

north of a95r0 is south of a96r0.

southwest of a97r0 is northeast of a98r0.

north of a99r0 is south of a100r0.

west of a101r0 is east of a102r0.

west of a103r0 is east of a104r0.

northwest of a5r0 is southeast of a105r0.

southeast of a34r1 is northwest of a106r1.

southeast of a36r1 is northwest of a107r1.

southwest of a37r1 is northeast of a108r1.

southwest of a38r1 is northeast of a109r1.

northwest of a31r1 is southeast of a110r1.

northeast of a3r0 is southwest of a111r0.

northwest of a4r0 is southeast of a112r0.

west of a0r0 is east of a94r0.

south of a2r0 is north of a20r0.

north of a0r0 is south of a6r0.

east of a7r0 is west of a96r0.

northeast of a7r0 is southwest of a98r0.

west of a23r0 is east of a100r0.

east of a9r0 is west of a102r0.

east of a11r0 is west of a104r0.

west of a12r0 is east of a76r0.

southeast of a24r0 is northwest of a50r0.

southwest of a19r0 is northeast of a52r0.

southwest of a21r0 is northeast of a54r0.

southwest of a14r0 is northeast of a56r0.

east of a22r0 is west of a58r0.

east of a17r0 is west of a60r0.

south of a1r0 is north of a18r0.

west of a20r0 is east of a92r0.

west of a42r1 is east of a68r1.

northeast of a28r1 is southwest of a70r1.

northwest of a29r1 is southeast of a72r1.

northwest of a27r1 is southeast of a74r1.

east of a30r1 is west of a40r1.

southwest of a25r1 is northeast of a40r1.

southeast of a48r1 is northwest of a113r1.

west of a47r1 is east of a88r1.

west of a46r1 is east of a90r1.

west of a32r1 is east of a62r1.

east of a33r1 is west of a64r1.

west of a33r1 is east of a39r1.

north of a35r1 is south of a41r1.

east of a41r1 is west of a66r1.

northeast of a8r0 is southwest of a78r0.

northeast of a10r0 is southwest of a80r0.

northwest of a13r1 is southeast of a45r1.

southwest of a44r1 is northeast of a82r1.

northwest of a43r1 is southeast of a113r1.

west of a84r1 is east of a113r1.

southwest of a48r1 is northeast of a86r1.

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
the player is in a56r0.

the orientation of the player is 6.
