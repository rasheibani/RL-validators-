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
the X of the a0r0 is "545768.0".
the Y of the a0r0 is "1018991.0".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "545736.7862838916".
the Y of the a1r0 is "1019016.633971292".

[create a2r0]
a2r0 is a area. "An area (2) in r0".
the printed name of the a2r0 is "Room 0".
Understand "Area 2 in Room 0" as a2r0.
the parent of the a2r0 is r0.
the X of the a2r0 is "546281.0".
the Y of the a2r0 is "1018991.0".

[create a3r0]
a3r0 is a area. "An area (3) in r0".
the printed name of the a3r0 is "Room 0".
Understand "Area 3 in Room 0" as a3r0.
the parent of the a3r0 is r0.
the X of the a3r0 is "546312.2137161084".
the Y of the a3r0 is "1019016.633971292".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "547472.0".
the Y of the a6r0 is "1018991.0".

[create a7r0]
a7r0 is a area. "An area (7) in r0".
the printed name of the a7r0 is "Room 0".
Understand "Area 7 in Room 0" as a7r0.
the parent of the a7r0 is r0.
the X of the a7r0 is "547440.7862838916".
the Y of the a7r0 is "1019016.633971292".

[create a8r0]
a8r0 is a area. "An area (8) in r0".
the printed name of the a8r0 is "Room 0".
Understand "Area 8 in Room 0" as a8r0.
the parent of the a8r0 is r0.
the X of the a8r0 is "545000.0".
the Y of the a8r0 is "1019080.0".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "545040.2945277712".
the Y of the a9r0 is "1019061.5971994374".

[create a10r0]
a10r0 is a area. "An area (10) in r0".
the printed name of the a10r0 is "Room 0".
Understand "Area 10 in Room 0" as a10r0.
the parent of the a10r0 is r0.
the X of the a10r0 is "545777.0".
the Y of the a10r0 is "1019080.0".

[create a11r0]
a11r0 is a area. "An area (11) in r0".
the printed name of the a11r0 is "Room 0".
Understand "Area 11 in Room 0" as a11r0.
the parent of the a11r0 is r0.
the X of the a11r0 is "545736.7054722288".
the Y of the a11r0 is "1019061.5971994377".

[create a12r0]
a12r0 is a area. "An area (12) in r0".
the printed name of the a12r0 is "Room 0".
Understand "Area 12 in Room 0" as a12r0.
the parent of the a12r0 is r0.
the X of the a12r0 is "546272.0".
the Y of the a12r0 is "1019080.0".

[create a13r0]
a13r0 is a area. "An area (13) in r0".
the printed name of the a13r0 is "Room 0".
Understand "Area 13 in Room 0" as a13r0.
the parent of the a13r0 is r0.
the X of the a13r0 is "546313.3556858187".
the Y of the a13r0 is "1019061.7045075548".

[create a14r0]
a14r0 is a area. "An area (14) in r0".
the printed name of the a14r0 is "Room 0".
Understand "Area 14 in Room 0" as a14r0.
the parent of the a14r0 is r0.
the X of the a14r0 is "547481.0".
the Y of the a14r0 is "1019080.0".

[create a15r0]
a15r0 is a area. "An area (15) in r0".
the printed name of the a15r0 is "Room 0".
Understand "Area 15 in Room 0" as a15r0.
the parent of the a15r0 is r0.
the X of the a15r0 is "547438.1485252038".
the Y of the a15r0 is "1019061.8557671143".

[create a20r0]
a20r0 is a area. "An area (20) in r0".
the printed name of the a20r0 is "Room 0".
Understand "Area 20 in Room 0" as a20r0.
the parent of the a20r0 is r0.
the X of the a20r0 is "545009.0".
the Y of the a20r0 is "1018991.0".

[create a21r0]
a21r0 is a area. "An area (21) in r0".
the printed name of the a21r0 is "Room 0".
Understand "Area 21 in Room 0" as a21r0.
the parent of the a21r0 is r0.
the X of the a21r0 is "545040.2137161086".
the Y of the a21r0 is "1019016.633971292".

[create a46r0]
a46r0 is a area. "An area (46) in r0".
the printed name of the a46r0 is "Room 0".
Understand "Area 46 in Room 0" as a46r0.
the parent of the a46r0 is r0.
the X of the a46r0 is "546629.0898876404".
the Y of the a46r0 is "1019000.0".

[create a47r0]
a47r0 is a area. "An area (47) in r0".
the printed name of the a47r0 is "Room 0".
Understand "Area 47 in Room 0" as a47r0.
the parent of the a47r0 is r0.
the X of the a47r0 is "546662.0417059885".
the Y of the a47r0 is "1018998.7861894328".

[create a50r0]
a50r0 is a area. "An area (50) in r0".
the printed name of the a50r0 is "Room 0".
Understand "Area 50 in Room 0" as a50r0.
the parent of the a50r0 is r0.
the X of the a50r0 is "547123.9101123596".
the Y of the a50r0 is "1019000.0".

[create a51r0]
a51r0 is a area. "An area (51) in r0".
the printed name of the a51r0 is "Room 0".
Understand "Area 51 in Room 0" as a51r0.
the parent of the a51r0 is r0.
the X of the a51r0 is "547090.9582940115".
the Y of the a51r0 is "1018998.7861894328".

[create a73r0]
a73r0 is a area. "An area (73) in r0".
the printed name of the a73r0 is "Room 0".
Understand "Area 73 in Room 0" as a73r0.
the parent of the a73r0 is r0.
the X of the a73r0 is "547423.7817693062".
the Y of the a73r0 is "1019044.9748289345".

[create a75r0]
a75r0 is a area. "An area (75) in r0".
the printed name of the a75r0 is "Room 0".
Understand "Area 75 in Room 0" as a75r0.
the parent of the a75r0 is r0.
the X of the a75r0 is "546674.1952020759".
the Y of the a75r0 is "1019131.6173773187".

[create a77r0]
a77r0 is a area. "An area (77) in r0".
the printed name of the a77r0 is "Room 0".
Understand "Area 77 in Room 0" as a77r0.
the parent of the a77r0 is r0.
the X of the a77r0 is "547077.8898716378".
the Y of the a77r0 is "1019141.6170304696".

[create a78r0]
a78r0 is a area. "An area (78) in r0".
the printed name of the a78r0 is "Room 0".
Understand "Area 78 in Room 0" as a78r0.
the parent of the a78r0 is r0.
the X of the a78r0 is "546710.0576256139".
the Y of the a78r0 is "1019161.865977651".

[create a79r0]
a79r0 is a area. "An area (79) in r0".
the printed name of the a79r0 is "Room 0".
Understand "Area 79 in Room 0" as a79r0.
the parent of the a79r0 is r0.
the X of the a79r0 is "547057.4535442034".
the Y of the a79r0 is "1019162.6384470325".

[create a81r0]
a81r0 is a area. "An area (81) in r0".
the printed name of the a81r0 is "Room 0".
Understand "Area 81 in Room 0" as a81r0.
the parent of the a81r0 is r0.
the X of the a81r0 is "545056.3340978592".
the Y of the a81r0 is "1019043.5012742098".

[create a82r0]
a82r0 is a area. "An area (82) in r0".
the printed name of the a82r0 is "Room 0".
Understand "Area 82 in Room 0" as a82r0.
the parent of the a82r0 is r0.
the X of the a82r0 is "545720.6659021408".
the Y of the a82r0 is "1019043.50127421".

[create a83r0]
a83r0 is a area. "An area (83) in r0".
the printed name of the a83r0 is "Room 0".
Understand "Area 83 in Room 0" as a83r0.
the parent of the a83r0 is r0.
the X of the a83r0 is "546328.634469697".
the Y of the a83r0 is "1019044.0018939395".

[create a89r0]
a89r0 is a area. "An area (89) in r0".
the printed name of the a89r0 is "Room 0".
Understand "Area 89 in Room 0" as a89r0.
the parent of the a89r0 is r0.
the X of the a89r0 is "547077.6602166065".
the Y of the a89r0 is "1020015.0".

[create a92r0]
a92r0 is a area. "An area (92) in r0".
the printed name of the a92r0 is "Room 0".
Understand "Area 92 in Room 0" as a92r0.
the parent of the a92r0 is r0.
the X of the a92r0 is "545383.3961068324".
the Y of the a92r0 is "1019128.733816152".

[create a93r0]
a93r0 is a area. "An area (93) in r0".
the printed name of the a93r0 is "Room 0".
Understand "Area 93 in Room 0" as a93r0.
the parent of the a93r0 is r0.
the X of the a93r0 is "545604.9376088154".
the Y of the a93r0 is "1019842.9379071898".

[create a95r0]
a95r0 is a area. "An area (95) in r0".
the printed name of the a95r0 is "Room 0".
Understand "Area 95 in Room 0" as a95r0.
the parent of the a95r0 is r0.
the X of the a95r0 is "546472.9076679605".
the Y of the a95r0 is "1019856.9294065498".

[create a36r1]
a36r1 is a area. "An area (36) in r1".
the printed name of the a36r1 is "Room 1".
Understand "Area 36 in Room 1" as a36r1.
the parent of the a36r1 is r1.
the X of the a36r1 is "545888.0".
the Y of the a36r1 is "1021101.0".

[create a37r1]
a37r1 is a area. "An area (37) in r1".
the printed name of the a37r1 is "Room 1".
Understand "Area 37 in Room 1" as a37r1.
the parent of the a37r1 is r1.
the X of the a37r1 is "545928.1595627206".
the Y of the a37r1 is "1021046.4402178596".

[create a62r1]
a62r1 is a area. "An area (62) in r1".
the printed name of the a62r1 is "Room 1".
Understand "Area 62 in Room 1" as a62r1.
the parent of the a62r1 is r1.
the X of the a62r1 is "546155.0".
the Y of the a62r1 is "1021132.0".

[create a63r1]
a63r1 is a area. "An area (63) in r1".
the printed name of the a63r1 is "Room 1".
Understand "Area 63 in Room 1" as a63r1.
the parent of the a63r1 is r1.
the X of the a63r1 is "546127.259315724".
the Y of the a63r1 is "1021079.5971839245".

[create a72r1]
a72r1 is a area. "An area (72) in r1".
the printed name of the a72r1 is "Room 1".
Understand "Area 72 in Room 1" as a72r1.
the parent of the a72r1 is r1.
the X of the a72r1 is "546002.5871593204".
the Y of the a72r1 is "1020964.226515985".

[create a88r1]
a88r1 is a area. "An area (88) in r1".
the printed name of the a88r1 is "Room 1".
Understand "Area 88 in Room 1" as a88r1.
the parent of the a88r1 is r1.
the X of the a88r1 is "546054.7557274033".
the Y of the a88r1 is "1020963.3385288817".

[create a22r2]
a22r2 is a area. "An area (22) in r2".
the printed name of the a22r2 is "Room 2".
Understand "Area 22 in Room 2" as a22r2.
the parent of the a22r2 is r2.
the X of the a22r2 is "547463.0".
the Y of the a22r2 is "1021114.0".

[create a23r2]
a23r2 is a area. "An area (23) in r2".
the printed name of the a23r2 is "Room 2".
Understand "Area 23 in Room 2" as a23r2.
the parent of the a23r2 is r2.
the X of the a23r2 is "547431.9123931623".
the Y of the a23r2 is "1021088.1662393162".

[create a26r2]
a26r2 is a area. "An area (26) in r2".
the printed name of the a26r2 is "Room 2".
Understand "Area 26 in Room 2" as a26r2.
the parent of the a26r2 is r2.
the X of the a26r2 is "548561.0".
the Y of the a26r2 is "1021104.0".

[create a27r2]
a27r2 is a area. "An area (27) in r2".
the printed name of the a27r2 is "Room 2".
Understand "Area 27 in Room 2" as a27r2.
the parent of the a27r2 is r2.
the X of the a27r2 is "548528.005982906".
the Y of the a27r2 is "1021105.7505982905".

[create a28r2]
a28r2 is a area. "An area (28) in r2".
the printed name of the a28r2 is "Room 2".
Understand "Area 28 in Room 2" as a28r2.
the parent of the a28r2 is r2.
the X of the a28r2 is "547980.0721649484".
the Y of the a28r2 is "1021104.7216494846".

[create a29r2]
a29r2 is a area. "An area (29) in r2".
the printed name of the a29r2 is "Room 2".
Understand "Area 29 in Room 2" as a29r2.
the parent of the a29r2 is r2.
the X of the a29r2 is "548013.0127059654".
the Y of the a29r2 is "1021106.113162393".

[create a30r2]
a30r2 is a area. "An area (30) in r2".
the printed name of the a30r2 is "Room 2".
Understand "Area 30 in Room 2" as a30r2.
the parent of the a30r2 is r2.
the X of the a30r2 is "548804.0".
the Y of the a30r2 is "1021024.0".

[create a31r2]
a31r2 is a area. "An area (31) in r2".
the printed name of the a31r2 is "Room 2".
Understand "Area 31 in Room 2" as a31r2.
the parent of the a31r2 is r2.
the X of the a31r2 is "548763.5982564336".
the Y of the a31r2 is "1021042.6848256432".

[create a38r2]
a38r2 is a area. "An area (38) in r2".
the printed name of the a38r2 is "Room 2".
Understand "Area 38 in Room 2" as a38r2.
the parent of the a38r2 is r2.
the X of the a38r2 is "547801.0".
the Y of the a38r2 is "1021114.0".

[create a39r2]
a39r2 is a area. "An area (39) in r2".
the printed name of the a39r2 is "Room 2".
Understand "Area 39 in Room 2" as a39r2.
the parent of the a39r2 is r2.
the X of the a39r2 is "547829.5998379212".
the Y of the a39r2 is "1021088.4150162078".

[create a40r2]
a40r2 is a area. "An area (40) in r2".
the printed name of the a40r2 is "Room 2".
Understand "Area 40 in Room 2" as a40r2.
the parent of the a40r2 is r2.
the X of the a40r2 is "546652.0".
the Y of the a40r2 is "1021114.0".

[create a41r2]
a41r2 is a area. "An area (41) in r2".
the printed name of the a41r2 is "Room 2".
Understand "Area 41 in Room 2" as a41r2.
the parent of the a41r2 is r2.
the X of the a41r2 is "546682.9153780069".
the Y of the a41r2 is "1021088.1834621994".

[create a42r2]
a42r2 is a area. "An area (42) in r2".
the printed name of the a42r2 is "Room 2".
Understand "Area 42 in Room 2" as a42r2.
the parent of the a42r2 is r2.
the X of the a42r2 is "548795.0".
the Y of the a42r2 is "1021114.0".

[create a43r2]
a43r2 is a area. "An area (43) in r2".
the printed name of the a43r2 is "Room 2".
Understand "Area 43 in Room 2" as a43r2.
the parent of the a43r2 is r2.
the X of the a43r2 is "548764.0846219931".
the Y of the a43r2 is "1021088.1834621993".

[create a52r2]
a52r2 is a area. "An area (52) in r2".
the printed name of the a52r2 is "Room 2".
Understand "Area 52 in Room 2" as a52r2.
the parent of the a52r2 is r2.
the X of the a52r2 is "548233.4677466997".
the Y of the a52r2 is "1020890.6330690054".

[create a53r2]
a53r2 is a area. "An area (53) in r2".
the printed name of the a53r2 is "Room 2".
Understand "Area 53 in Room 2" as a53r2.
the parent of the a53r2 is r2.
the X of the a53r2 is "548252.5097823646".
the Y of the a53r2 is "1020855.9318963648".

[create a54r2]
a54r2 is a area. "An area (54) in r2".
the printed name of the a54r2 is "Room 2".
Understand "Area 54 in Room 2" as a54r2.
the parent of the a54r2 is r2.
the X of the a54r2 is "547792.0".
the Y of the a54r2 is "1021024.0".

[create a55r2]
a55r2 is a area. "An area (55) in r2".
the printed name of the a55r2 is "Room 2".
Understand "Area 55 in Room 2" as a55r2.
the parent of the a55r2 is r2.
the X of the a55r2 is "547835.1182354635".
the Y of the a55r2 is "1021042.4131764536".

[create a56r2]
a56r2 is a area. "An area (56) in r2".
the printed name of the a56r2 is "Room 2".
Understand "Area 56 in Room 2" as a56r2.
the parent of the a56r2 is r2.
the X of the a56r2 is "546643.0".
the Y of the a56r2 is "1021024.0".

[create a57r2]
a57r2 is a area. "An area (57) in r2".
the printed name of the a57r2 is "Room 2".
Understand "Area 57 in Room 2" as a57r2.
the parent of the a57r2 is r2.
the X of the a57r2 is "546685.06".
the Y of the a57r2 is "1021042.5189999999".

[create a58r2]
a58r2 is a area. "An area (58) in r2".
the printed name of the a58r2 is "Room 2".
Understand "Area 58 in Room 2" as a58r2.
the parent of the a58r2 is r2.
the X of the a58r2 is "547472.0".
the Y of the a58r2 is "1021024.0".

[create a59r2]
a59r2 is a area. "An area (59) in r2".
the printed name of the a59r2 is "Room 2".
Understand "Area 59 in Room 2" as a59r2.
the parent of the a59r2 is r2.
the X of the a59r2 is "547429.94".
the Y of the a59r2 is "1021042.519".

[create a64r2]
a64r2 is a area. "An area (64) in r2".
the printed name of the a64r2 is "Room 2".
Understand "Area 64 in Room 2" as a64r2.
the parent of the a64r2 is r2.
the X of the a64r2 is "546700.0926719009".
the Y of the a64r2 is "1021059.806572686".

[create a65r2]
a65r2 is a area. "An area (65) in r2".
the printed name of the a65r2 is "Room 2".
Understand "Area 65 in Room 2" as a65r2.
the parent of the a65r2 is r2.
the X of the a65r2 is "548747.6927655565".
the Y of the a65r2 is "1021061.1041153659".

[create a67r2]
a67r2 is a area. "An area (67) in r2".
the printed name of the a67r2 is "Room 2".
Understand "Area 67 in Room 2" as a67r2.
the parent of the a67r2 is r2.
the X of the a67r2 is "547414.8606168446".
the Y of the a67r2 is "1021059.8602906286".

[create a69r2]
a69r2 is a area. "An area (69) in r2".
the printed name of the a69r2 is "Room 2".
Understand "Area 69 in Room 2" as a69r2.
the parent of the a69r2 is r2.
the X of the a69r2 is "548520.4854185786".
the Y of the a69r2 is "1021031.6730396654".

[create a70r2]
a70r2 is a area. "An area (70) in r2".
the printed name of the a70r2 is "Room 2".
Understand "Area 70 in Room 2" as a70r2.
the parent of the a70r2 is r2.
the X of the a70r2 is "548018.3573792815".
the Y of the a70r2 is "1021047.1425703285".

[create a71r2]
a71r2 is a area. "An area (71) in r2".
the printed name of the a71r2 is "Room 2".
Understand "Area 71 in Room 2" as a71r2.
the parent of the a71r2 is r2.
the X of the a71r2 is "547848.9673953156".
the Y of the a71r2 is "1021058.7490639035".

[create a80r2]
a80r2 is a area. "An area (80) in r2".
the printed name of the a80r2 is "Room 2".
Understand "Area 80 in Room 2" as a80r2.
the parent of the a80r2 is r2.
the X of the a80r2 is "548383.6906465164".
the Y of the a80r2 is "1020900.8203483168".

[create a86r2]
a86r2 is a area. "An area (86) in r2".
the printed name of the a86r2 is "Room 2".
Understand "Area 86 in Room 2" as a86r2.
the parent of the a86r2 is r2.
the X of the a86r2 is "548378.6429331244".
the Y of the a86r2 is "1020948.302399796".

[create a87r2]
a87r2 is a area. "An area (87) in r2".
the printed name of the a87r2 is "Room 2".
Understand "Area 87 in Room 2" as a87r2.
the parent of the a87r2 is r2.
the X of the a87r2 is "548145.3216816066".
the Y of the a87r2 is "1020981.1731561061".

[create a90r2]
a90r2 is a area. "An area (90) in r2".
the printed name of the a90r2 is "Room 2".
Understand "Area 90 in Room 2" as a90r2.
the parent of the a90r2 is r2.
the X of the a90r2 is "547445.7347933755".
the Y of the a90r2 is "1020169.6178680473".

[create a91r2]
a91r2 is a area. "An area (91) in r2".
the printed name of the a91r2 is "Room 2".
Understand "Area 91 in Room 2" as a91r2.
the parent of the a91r2 is r2.
the X of the a91r2 is "547057.0".
the Y of the a91r2 is "1020935.337264151".

[create a16r3]
a16r3 is a area. "An area (16) in r3".
the printed name of the a16r3 is "Room 3".
Understand "Area 16 in Room 3" as a16r3.
the parent of the a16r3 is r3.
the X of the a16r3 is "549191.0".
the Y of the a16r3 is "1019000.0".

[create a17r3]
a17r3 is a area. "An area (17) in r3".
the printed name of the a17r3 is "Room 3".
Understand "Area 17 in Room 3" as a17r3.
the parent of the a17r3 is r3.
the X of the a17r3 is "549148.4517163367".
the Y of the a17r3 is "1019049.4252993808".

[create a60r3]
a60r3 is a area. "An area (60) in r3".
the printed name of the a60r3 is "Room 3".
Understand "Area 60 in Room 3" as a60r3.
the parent of the a60r3 is r3.
the X of the a60r3 is "548890.0".
the Y of the a60r3 is "1018972.0".

[create a61r3]
a61r3 is a area. "An area (61) in r3".
the printed name of the a61r3 is "Room 3".
Understand "Area 61 in Room 3" as a61r3.
the parent of the a61r3 is r3.
the X of the a61r3 is "548923.0639588251".
the Y of the a61r3 is "1019024.5311926303".

[create a74r3]
a74r3 is a area. "An area (74) in r3".
the printed name of the a74r3 is "Room 3".
Understand "Area 74 in Room 3" as a74r3.
the parent of the a74r3 is r3.
the X of the a74r3 is "549084.9629832634".
the Y of the a74r3 is "1019128.2779361999".

[create a94r3]
a94r3 is a area. "An area (94) in r3".
the printed name of the a94r3 is "Room 3".
Understand "Area 94 in Room 3" as a94r3.
the parent of the a94r3 is r3.
the X of the a94r3 is "548993.6483259039".
the Y of the a94r3 is "1019133.517625388".

[create a24r4]
a24r4 is a area. "An area (24) in r4".
the printed name of the a24r4 is "Room 4".
Understand "Area 24 in Room 4" as a24r4.
the parent of the a24r4 is r4.
the X of the a24r4 is "550101.0".
the Y of the a24r4 is "1021114.0".

[create a25r4]
a25r4 is a area. "An area (25) in r4".
the printed name of the a25r4 is "Room 4".
Understand "Area 25 in Room 4" as a25r4.
the parent of the a25r4 is r4.
the X of the a25r4 is "550069.9123931624".
the Y of the a25r4 is "1021088.1662393163".

[create a32r4]
a32r4 is a area. "An area (32) in r4".
the printed name of the a32r4 is "Room 4".
Understand "Area 32 in Room 4" as a32r4.
the parent of the a32r4 is r4.
the X of the a32r4 is "549333.0".
the Y of the a32r4 is "1021024.0".

[create a33r4]
a33r4 is a area. "An area (33) in r4".
the printed name of the a33r4 is "Room 4".
Understand "Area 33 in Room 4" as a33r4.
the parent of the a33r4 is r4.
the X of the a33r4 is "549373.7622699386".
the Y of the a33r4 is "1021042.648773006".

[create a34r4]
a34r4 is a area. "An area (34) in r4".
the printed name of the a34r4 is "Room 4".
Understand "Area 34 in Room 4" as a34r4.
the parent of the a34r4 is r4.
the X of the a34r4 is "550110.0".
the Y of the a34r4 is "1021024.0".

[create a35r4]
a35r4 is a area. "An area (35) in r4".
the printed name of the a35r4 is "Room 4".
Understand "Area 35 in Room 4" as a35r4.
the parent of the a35r4 is r4.
the X of the a35r4 is "550071.1323529411".
the Y of the a35r4 is "1021042.8382352941".

[create a44r4]
a44r4 is a area. "An area (44) in r4".
the printed name of the a44r4 is "Room 4".
Understand "Area 44 in Room 4" as a44r4.
the parent of the a44r4 is r4.
the X of the a44r4 is "549342.0".
the Y of the a44r4 is "1021114.0".

[create a45r4]
a45r4 is a area. "An area (45) in r4".
the printed name of the a45r4 is "Room 4".
Understand "Area 45 in Room 4" as a45r4.
the parent of the a45r4 is r4.
the X of the a45r4 is "549373.0876068375".
the Y of the a45r4 is "1021088.1662393163".

[create a66r4]
a66r4 is a area. "An area (66) in r4".
the printed name of the a66r4 is "Room 4".
Understand "Area 66 in Room 4" as a66r4.
the parent of the a66r4 is r4.
the X of the a66r4 is "549389.5405048298".
the Y of the a66r4 is "1021060.8544286496".

[create a68r4]
a68r4 is a area. "An area (68) in r4".
the printed name of the a68r4 is "Room 4".
Understand "Area 68 in Room 4" as a68r4.
the parent of the a68r4 is r4.
the X of the a68r4 is "550053.8307569795".
the Y of the a68r4 is "1021061.4707232527".

[create a85r4]
a85r4 is a area. "An area (85) in r4".
the printed name of the a85r4 is "Room 4".
Understand "Area 85 in Room 4" as a85r4.
the parent of the a85r4 is r4.
the X of the a85r4 is "549729.4895726717".
the Y of the a85r4 is "1020972.4549042069".

[create a4r5]
a4r5 is a area. "An area (4) in r5".
the printed name of the a4r5 is "Room 5".
Understand "Area 4 in Room 5" as a4r5.
the parent of the a4r5 is r5.
the X of the a4r5 is "548667.0".
the Y of the a4r5 is "1019080.0".

[create a5r5]
a5r5 is a area. "An area (5) in r5".
the printed name of the a5r5 is "Room 5".
Understand "Area 5 in Room 5" as a5r5.
the parent of the a5r5 is r5.
the X of the a5r5 is "548634.5645125471".
the Y of the a5r5 is "1019060.8024650232".

[create a18r5]
a18r5 is a area. "An area (18) in r5".
the printed name of the a18r5 is "Room 5".
Understand "Area 18 in Room 5" as a18r5.
the parent of the a18r5 is r5.
the X of the a18r5 is "548658.0".
the Y of the a18r5 is "1018991.0".

[create a19r5]
a19r5 is a area. "An area (19) in r5".
the printed name of the a19r5 is "Room 5".
Understand "Area 19 in Room 5" as a19r5.
the parent of the a19r5 is r5.
the X of the a19r5 is "548620.3333034967".
the Y of the a19r5 is "1019017.2865198712".

[create a48r5]
a48r5 is a area. "An area (48) in r5".
the printed name of the a48r5 is "Room 5".
Understand "Area 48 in Room 5" as a48r5.
the parent of the a48r5 is r5.
the X of the a48r5 is "548145.0".
the Y of the a48r5 is "1018972.0".

[create a49r5]
a49r5 is a area. "An area (49) in r5".
the printed name of the a49r5 is "Room 5".
Understand "Area 49 in Room 5" as a49r5.
the parent of the a49r5 is r5.
the X of the a49r5 is "548178.978451452".
the Y of the a49r5 is "1019045.2960965103".

[create a76r5]
a76r5 is a area. "An area (76) in r5".
the printed name of the a76r5 is "Room 5".
Understand "Area 76 in Room 5" as a76r5.
the parent of the a76r5 is r5.
the X of the a76r5 is "548222.6494994545".
the Y of the a76r5 is "1019130.2617805512".

[create a84r5]
a84r5 is a area. "An area (84) in r5".
the printed name of the a84r5 is "Room 5".
Understand "Area 84 in Room 5" as a84r5.
the parent of the a84r5 is r5.
the X of the a84r5 is "548610.6123477007".
the Y of the a84r5 is "1019033.3018313111".

[create d0]
d0 is a door. "A door between a74r3 and a85r4".
d0 is north of a74r3 and south of a85r4.

[create d2]
d2 is a door. "A door between a90r2 and a76r5".
d2 is southeast of a90r2 and northwest of a76r5.

[create d4]
d4 is a door. "A door between a80r2 and a94r3".
d4 is south of a80r2 and north of a94r3.

[create d6]
d6 is a door. "A door between a95r0 and a88r1".
d6 is north of a95r0 and south of a88r1.

[create d7]
d7 is a door. "A door between a93r0 and a72r1".
d7 is north of a93r0 and south of a72r1.

[create d10]
d10 is a door. "A door between a89r0 and a90r2".
d10 is northeast of a89r0 and southwest of a90r2.

[create d11]
d11 is a door. "A door between a89r0 and a91r2".
d11 is north of a89r0 and south of a91r2.

northwest of a0r0 is southeast of a1r0.

northeast of a2r0 is southwest of a3r0.

southwest of a4r5 is northeast of a5r5.

northwest of a6r0 is southeast of a7r0.

southeast of a8r0 is northwest of a9r0.

southwest of a10r0 is northeast of a11r0.

southeast of a12r0 is northwest of a13r0.

southwest of a14r0 is northeast of a15r0.

northwest of a16r3 is southeast of a17r3.

northwest of a18r5 is southeast of a19r5.

northeast of a20r0 is southwest of a21r0.

southwest of a22r2 is northeast of a23r2.

southwest of a24r4 is northeast of a25r4.

west of a26r2 is east of a27r2.

east of a28r2 is west of a29r2.

northwest of a30r2 is southeast of a31r2.

northeast of a32r4 is southwest of a33r4.

northwest of a34r4 is southeast of a35r4.

southeast of a36r1 is northwest of a37r1.

southeast of a38r2 is northwest of a39r2.

southeast of a40r2 is northwest of a41r2.

southwest of a42r2 is northeast of a43r2.

southeast of a44r4 is northwest of a45r4.

east of a46r0 is west of a47r0.

northeast of a48r5 is southwest of a49r5.

west of a50r0 is east of a51r0.

southeast of a52r2 is northwest of a53r2.

northeast of a54r2 is southwest of a55r2.

northeast of a56r2 is southwest of a57r2.

northwest of a58r2 is southeast of a59r2.

northeast of a60r3 is southwest of a61r3.

southwest of a62r1 is northeast of a63r1.

southeast of a41r2 is northwest of a64r2.

southwest of a43r2 is northeast of a65r2.

southeast of a45r4 is northwest of a66r4.

southwest of a23r2 is northeast of a67r2.

southwest of a25r4 is northeast of a68r4.

south of a27r2 is north of a69r2.

south of a29r2 is north of a70r2.

northeast of a55r2 is southwest of a71r2.

northeast of a57r2 is southwest of a64r2.

northwest of a59r2 is southeast of a67r2.

northwest of a31r2 is southeast of a65r2.

northeast of a33r4 is southwest of a66r4.

northwest of a35r4 is southeast of a68r4.

southeast of a37r1 is northwest of a72r1.

southeast of a39r2 is northwest of a71r2.

southwest of a15r0 is northeast of a73r0.

northwest of a17r3 is southeast of a74r3.

north of a47r0 is south of a75r0.

northeast of a49r5 is southwest of a76r5.

north of a51r0 is south of a77r0.

northeast of a75r0 is southwest of a78r0.

northwest of a77r0 is southeast of a79r0.

east of a53r2 is west of a80r2.

northeast of a21r0 is southwest of a81r0.

northwest of a1r0 is southeast of a82r0.

northeast of a3r0 is southwest of a83r0.

southwest of a5r5 is northeast of a84r5.

northwest of a7r0 is southeast of a73r0.

southeast of a9r0 is northwest of a81r0.

southwest of a11r0 is northeast of a82r0.

southeast of a13r0 is northwest of a83r0.

northwest of a19r5 is southeast of a84r5.

east of a66r4 is west of a85r4.

southwest of a69r2 is northeast of a86r2.

southeast of a70r2 is northwest of a87r2.

west of a68r4 is east of a85r4.

west of a70r2 is east of a71r2.

east of a72r1 is west of a88r1.

southwest of a63r1 is northeast of a88r1.

southwest of a87r2 is northeast of a90r2.

north of a80r2 is south of a86r2.

west of a67r2 is east of a91r2.

east of a64r2 is west of a91r2.

west of a65r2 is east of a69r2.

west of a86r2 is east of a87r2.

north of a92r0 is south of a93r0.

north of a79r0 is south of a89r0.

north of a78r0 is south of a95r0.

east of a93r0 is west of a95r0.

west of a73r0 is east of a77r0.

west of a75r0 is east of a83r0.

east of a81r0 is west of a92r0.

west of a82r0 is east of a92r0.

east of a78r0 is west of a79r0.

west of a74r3 is east of a94r3.

northeast of a61r3 is southwest of a94r3.

east of a76r5 is west of a84r5.

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
the player is in a5r5.

the orientation of the player is 6.
