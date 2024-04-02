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

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "546634.5383292555".
the Y of the a22r0 is "1733495.769022003".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "546713.6436566666".
the Y of the a23r0 is "1733562.334371083".

[create a24r0]
a24r0 is a area. "An area (24) in r0".
the printed name of the a24r0 is "Room 0".
Understand "Area 24 in Room 0" as a24r0.
the parent of the a24r0 is r0.
the X of the a24r0 is "545394.8633422253".
the Y of the a24r0 is "1733534.3385130365".

[create a25r0]
a25r0 is a area. "An area (25) in r0".
the printed name of the a25r0 is "Room 0".
Understand "Area 25 in Room 0" as a25r0.
the parent of the a25r0 is r0.
the X of the a25r0 is "546093.0873366152".
the Y of the a25r0 is "1734449.1596503141".

[create a26r0]
a26r0 is a area. "An area (26) in r0".
the printed name of the a26r0 is "Room 0".
Understand "Area 26 in Room 0" as a26r0.
the parent of the a26r0 is r0.
the X of the a26r0 is "546958.5439185295".
the Y of the a26r0 is "1733542.5987078769".

[create a31r0]
a31r0 is a area. "An area (31) in r0".
the printed name of the a31r0 is "Room 0".
Understand "Area 31 in Room 0" as a31r0.
the parent of the a31r0 is r0.
the X of the a31r0 is "545743.0171235859".
the Y of the a31r0 is "1733442.915905135".

[create a32r0]
a32r0 is a area. "An area (32) in r0".
the printed name of the a32r0 is "Room 0".
Understand "Area 32 in Room 0" as a32r0.
the parent of the a32r0 is r0.
the X of the a32r0 is "545077.0941579642".
the Y of the a32r0 is "1733441.5825458067".

[create a35r0]
a35r0 is a area. "An area (35) in r0".
the printed name of the a35r0 is "Room 0".
Understand "Area 35 in Room 0" as a35r0.
the parent of the a35r0 is r0.
the X of the a35r0 is "547090.988395043".
the Y of the a35r0 is "1733464.0339803433".

[create a39r0]
a39r0 is a area. "An area (39) in r0".
the printed name of the a39r0 is "Room 0".
Understand "Area 39 in Room 0" as a39r0.
the parent of the a39r0 is r0.
the X of the a39r0 is "547327.8228866081".
the Y of the a39r0 is "1733440.0615040062".

[create a42r0]
a42r0 is a area. "An area (42) in r0".
the printed name of the a42r0 is "Room 0".
Understand "Area 42 in Room 0" as a42r0.
the parent of the a42r0 is r0.
the X of the a42r0 is "546335.0670611437".
the Y of the a42r0 is "1733444.7228796843".

[create a43r0]
a43r0 is a area. "An area (43) in r0".
the printed name of the a43r0 is "Room 0".
Understand "Area 43 in Room 0" as a43r0.
the parent of the a43r0 is r0.
the X of the a43r0 is "547373.0".
the Y of the a43r0 is "1733391.0".

[create a44r0]
a44r0 is a area. "An area (44) in r0".
the printed name of the a44r0 is "Room 0".
Understand "Area 44 in Room 0" as a44r0.
the parent of the a44r0 is r0.
the X of the a44r0 is "547341.9579422104".
the Y of the a44r0 is "1733416.6166125853".

[create a63r0]
a63r0 is a area. "An area (63) in r0".
the printed name of the a63r0 is "Room 0".
Understand "Area 63 in Room 0" as a63r0.
the parent of the a63r0 is r0.
the X of the a63r0 is "547129.0".
the Y of the a63r0 is "1733391.0".

[create a64r0]
a64r0 is a area. "An area (64) in r0".
the printed name of the a64r0 is "Room 0".
Understand "Area 64 in Room 0" as a64r0.
the parent of the a64r0 is r0.
the X of the a64r0 is "547096.9582940115".
the Y of the a64r0 is "1733398.7861894327".

[create a65r0]
a65r0 is a area. "An area (65) in r0".
the printed name of the a65r0 is "Room 0".
Understand "Area 65 in Room 0" as a65r0.
the parent of the a65r0 is r0.
the X of the a65r0 is "546596.0".
the Y of the a65r0 is "1733391.0".

[create a66r0]
a66r0 is a area. "An area (66) in r0".
the printed name of the a66r0 is "Room 0".
Understand "Area 66 in Room 0" as a66r0.
the parent of the a66r0 is r0.
the X of the a66r0 is "546627.8650009037".
the Y of the a66r0 is "1733398.768320379".

[create a77r0]
a77r0 is a area. "An area (77) in r0".
the printed name of the a77r0 is "Room 0".
Understand "Area 77 in Room 0" as a77r0.
the parent of the a77r0 is r0.
the X of the a77r0 is "546287.0".
the Y of the a77r0 is "1733391.0".

[create a78r0]
a78r0 is a area. "An area (78) in r0".
the printed name of the a78r0 is "Room 0".
Understand "Area 78 in Room 0" as a78r0.
the parent of the a78r0 is r0.
the X of the a78r0 is "546318.2137161085".
the Y of the a78r0 is "1733416.6339712923".

[create a79r0]
a79r0 is a area. "An area (79) in r0".
the printed name of the a79r0 is "Room 0".
Understand "Area 79 in Room 0" as a79r0.
the parent of the a79r0 is r0.
the X of the a79r0 is "545022.0".
the Y of the a79r0 is "1733480.0".

[create a80r0]
a80r0 is a area. "An area (80) in r0".
the printed name of the a80r0 is "Room 0".
Understand "Area 80 in Room 0" as a80r0.
the parent of the a80r0 is r0.
the X of the a80r0 is "545058.0183625785".
the Y of the a80r0 is "1733461.1647782384".

[create a81r0]
a81r0 is a area. "An area (81) in r0".
the printed name of the a81r0 is "Room 0".
Understand "Area 81 in Room 0" as a81r0.
the parent of the a81r0 is r0.
the X of the a81r0 is "545799.0".
the Y of the a81r0 is "1733480.0".

[create a82r0]
a82r0 is a area. "An area (82) in r0".
the printed name of the a82r0 is "Room 0".
Understand "Area 82 in Room 0" as a82r0.
the parent of the a82r0 is r0.
the X of the a82r0 is "545760.2797213208".
the Y of the a82r0 is "1733461.4380057093".

[create a89r0]
a89r0 is a area. "An area (89) in r0".
the printed name of the a89r0 is "Room 0".
Understand "Area 89 in Room 0" as a89r0.
the parent of the a89r0 is r0.
the X of the a89r0 is "546278.0".
the Y of the a89r0 is "1733480.0".

[create a90r0]
a90r0 is a area. "An area (90) in r0".
the printed name of the a90r0 is "Room 0".
Understand "Area 90 in Room 0" as a90r0.
the parent of the a90r0 is r0.
the X of the a90r0 is "546320.2240682535".
the Y of the a90r0 is "1733461.7923215088".

[create a95r0]
a95r0 is a area. "An area (95) in r0".
the printed name of the a95r0 is "Room 0".
Understand "Area 95 in Room 0" as a95r0.
the parent of the a95r0 is r0.
the X of the a95r0 is "545031.0".
the Y of the a95r0 is "1733391.0".

[create a96r0]
a96r0 is a area. "An area (96) in r0".
the printed name of the a96r0 is "Room 0".
Understand "Area 96 in Room 0" as a96r0.
the parent of the a96r0 is r0.
the X of the a96r0 is "545062.0420577896".
the Y of the a96r0 is "1733416.6166125855".

[create a97r0]
a97r0 is a area. "An area (97) in r0".
the printed name of the a97r0 is "Room 0".
Understand "Area 97 in Room 0" as a97r0.
the parent of the a97r0 is r0.
the X of the a97r0 is "547382.0".
the Y of the a97r0 is "1733480.0".

[create a98r0]
a98r0 is a area. "An area (98) in r0".
the printed name of the a98r0 is "Room 0".
Understand "Area 98 in Room 0" as a98r0.
the parent of the a98r0 is r0.
the X of the a98r0 is "547354.849158728".
the Y of the a98r0 is "1733460.2680626004".

[create a99r0]
a99r0 is a area. "An area (99) in r0".
the printed name of the a99r0 is "Room 0".
Understand "Area 99 in Room 0" as a99r0.
the parent of the a99r0 is r0.
the X of the a99r0 is "545790.0".
the Y of the a99r0 is "1733391.0".

[create a100r0]
a100r0 is a area. "An area (100) in r0".
the printed name of the a100r0 is "Room 0".
Understand "Area 100 in Room 0" as a100r0.
the parent of the a100r0 is r0.
the X of the a100r0 is "545758.7862838916".
the Y of the a100r0 is "1733416.633971292".

[create a0r1]
a0r1 is a area. "An area (0) in r1".
the printed name of the a0r1 is "Room 1".
Understand "Area 0 in Room 1" as a0r1.
the parent of the a0r1 is r1.
the X of the a0r1 is "546709.8580843881".
the Y of the a0r1 is "1735374.299585809".

[create a1r1]
a1r1 is a area. "An area (1) in r1".
the printed name of the a1r1 is "Room 1".
Understand "Area 1 in Room 1" as a1r1.
the parent of the a1r1 is r1.
the X of the a1r1 is "547025.9357867723".
the Y of the a1r1 is "1735377.0091663753".

[create a2r1]
a2r1 is a area. "An area (2) in r1".
the printed name of the a2r1 is "Room 1".
Understand "Area 2 in Room 1" as a2r1.
the parent of the a2r1 is r1.
the X of the a2r1 is "546358.9589252132".
the Y of the a2r1 is "1735461.6795222147".

[create a3r1]
a3r1 is a area. "An area (3) in r1".
the printed name of the a3r1 is "Room 1".
Understand "Area 3 in Room 1" as a3r1.
the parent of the a3r1 is r1.
the X of the a3r1 is "546701.4209661519".
the Y of the a3r1 is "1735383.3451500714".

[create a9r1]
a9r1 is a area. "An area (9) in r1".
the printed name of the a9r1 is "Room 1".
Understand "Area 9 in Room 1" as a9r1.
the parent of the a9r1 is r1.
the X of the a9r1 is "547075.2223991761".
the Y of the a9r1 is "1735325.6120157628".

[create a10r1]
a10r1 is a area. "An area (10) in r1".
the printed name of the a10r1 is "Room 1".
Understand "Area 10 in Room 1" as a10r1.
the parent of the a10r1 is r1.
the X of the a10r1 is "547098.0735241217".
the Y of the a10r1 is "1735211.726733836".

[create a11r1]
a11r1 is a area. "An area (11) in r1".
the printed name of the a11r1 is "Room 1".
Understand "Area 11 in Room 1" as a11r1.
the parent of the a11r1 is r1.
the X of the a11r1 is "545503.5427972078".
the Y of the a11r1 is "1735338.798521903".

[create a12r1]
a12r1 is a area. "An area (12) in r1".
the printed name of the a12r1 is "Room 1".
Understand "Area 12 in Room 1" as a12r1.
the parent of the a12r1 is r1.
the X of the a12r1 is "545883.9343690765".
the Y of the a12r1 is "1735461.642719334".

[create a13r1]
a13r1 is a area. "An area (13) in r1".
the printed name of the a13r1 is "Room 1".
Understand "Area 13 in Room 1" as a13r1.
the parent of the a13r1 is r1.
the X of the a13r1 is "547162.0000000001".
the Y of the a13r1 is "1735449.759326711".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "545135.4913012546".
the Y of the a14r1 is "1735462.596106584".

[create a21r1]
a21r1 is a area. "An area (21) in r1".
the printed name of the a21r1 is "Room 1".
Understand "Area 21 in Room 1" as a21r1.
the parent of the a21r1 is r1.
the X of the a21r1 is "546080.6892610956".
the Y of the a21r1 is "1734510.9362679888".

[create a45r1]
a45r1 is a area. "An area (45) in r1".
the printed name of the a45r1 is "Room 1".
Understand "Area 45 in Room 1" as a45r1.
the parent of the a45r1 is r1.
the X of the a45r1 is "545080.0".
the Y of the a45r1 is "1735424.0".

[create a46r1]
a46r1 is a area. "An area (46) in r1".
the printed name of the a46r1 is "Room 1".
Understand "Area 46 in Room 1" as a46r1.
the parent of the a46r1 is r1.
the X of the a46r1 is "545116.9075342466".
the Y of the a46r1 is "1735443.0342465756".

[create a49r1]
a49r1 is a area. "An area (49) in r1".
the printed name of the a49r1 is "Room 1".
Understand "Area 49 in Room 1" as a49r1.
the parent of the a49r1 is r1.
the X of the a49r1 is "546312.0".
the Y of the a49r1 is "1735514.0".

[create a50r1]
a50r1 is a area. "An area (50) in r1".
the printed name of the a50r1 is "Room 1".
Understand "Area 50 in Room 1" as a50r1.
the parent of the a50r1 is r1.
the X of the a50r1 is "546342.9153780069".
the Y of the a50r1 is "1735488.1834621995".

[create a51r1]
a51r1 is a area. "An area (51) in r1".
the printed name of the a51r1 is "Room 1".
Understand "Area 51 in Room 1" as a51r1.
the parent of the a51r1 is r1.
the X of the a51r1 is "545089.0".
the Y of the a51r1 is "1735514.0".

[create a52r1]
a52r1 is a area. "An area (52) in r1".
the printed name of the a52r1 is "Room 1".
Understand "Area 52 in Room 1" as a52r1.
the parent of the a52r1 is r1.
the X of the a52r1 is "545120.0876068377".
the Y of the a52r1 is "1735488.1662393163".

[create a53r1]
a53r1 is a area. "An area (53) in r1".
the printed name of the a53r1 is "Room 1".
Understand "Area 53 in Room 1" as a53r1.
the parent of the a53r1 is r1.
the X of the a53r1 is "545931.0".
the Y of the a53r1 is "1735514.0".

[create a54r1]
a54r1 is a area. "An area (54) in r1".
the printed name of the a54r1 is "Room 1".
Understand "Area 54 in Room 1" as a54r1.
the parent of the a54r1 is r1.
the X of the a54r1 is "545899.9123931625".
the Y of the a54r1 is "1735488.1662393163".

[create a57r1]
a57r1 is a area. "An area (57) in r1".
the printed name of the a57r1 is "Room 1".
Understand "Area 57 in Room 1" as a57r1.
the parent of the a57r1 is r1.
the X of the a57r1 is "546656.0".
the Y of the a57r1 is "1735504.0".

[create a58r1]
a58r1 is a area. "An area (58) in r1".
the printed name of the a58r1 is "Room 1".
Understand "Area 58 in Room 1" as a58r1.
the parent of the a58r1 is r1.
the X of the a58r1 is "546688.9940170939".
the Y of the a58r1 is "1735505.7505982905".

[create a59r1]
a59r1 is a area. "An area (59) in r1".
the printed name of the a59r1 is "Room 1".
Understand "Area 59 in Room 1" as a59r1.
the parent of the a59r1 is r1.
the X of the a59r1 is "547071.0".
the Y of the a59r1 is "1735514.0".

[create a60r1]
a60r1 is a area. "An area (60) in r1".
the printed name of the a60r1 is "Room 1".
Understand "Area 60 in Room 1" as a60r1.
the parent of the a60r1 is r1.
the X of the a60r1 is "547039.005982906".
the Y of the a60r1 is "1735505.7505982907".

[create a73r1]
a73r1 is a area. "An area (73) in r1".
the printed name of the a73r1 is "Room 1".
Understand "Area 73 in Room 1" as a73r1.
the parent of the a73r1 is r1.
the X of the a73r1 is "546303.0".
the Y of the a73r1 is "1735424.0".

[create a74r1]
a74r1 is a area. "An area (74) in r1".
the printed name of the a74r1 is "Room 1".
Understand "Area 74 in Room 1" as a74r1.
the parent of the a74r1 is r1.
the X of the a74r1 is "546341.6344476744".
the Y of the a74r1 is "1735442.8615552324".

[create a75r1]
a75r1 is a area. "An area (75) in r1".
the printed name of the a75r1 is "Room 1".
Understand "Area 75 in Room 1" as a75r1.
the parent of the a75r1 is r1.
the X of the a75r1 is "545940.0".
the Y of the a75r1 is "1735424.0".

[create a76r1]
a76r1 is a area. "An area (76) in r1".
the printed name of the a76r1 is "Room 1".
Understand "Area 76 in Room 1" as a76r1.
the parent of the a76r1 is r1.
the X of the a76r1 is "545901.0502336449".
the Y of the a76r1 is "1735442.8300233649".

[create a111r1]
a111r1 is a area. "An area (111) in r1".
the printed name of the a111r1 is "Room 1".
Understand "Area 111 in Room 1" as a111r1.
the parent of the a111r1 is r1.
the X of the a111r1 is "547192.0".
the Y of the a111r1 is "1735504.0".

[create a27r2]
a27r2 is a area. "An area (27) in r2".
the printed name of the a27r2 is "Room 2".
Understand "Area 27 in Room 2" as a27r2.
the parent of the a27r2 is r2.
the X of the a27r2 is "547873.7606867261".
the Y of the a27r2 is "1733549.586681897".

[create a36r2]
a36r2 is a area. "An area (36) in r2".
the printed name of the a36r2 is "Room 2".
Understand "Area 36 in Room 2" as a36r2.
the parent of the a36r2 is r2.
the X of the a36r2 is "547808.5661679944".
the Y of the a36r2 is "1733419.9341592467".

[create a37r2]
a37r2 is a area. "An area (37) in r2".
the printed name of the a37r2 is "Room 2".
Understand "Area 37 in Room 2" as a37r2.
the parent of the a37r2 is r2.
the X of the a37r2 is "547871.0091290363".
the Y of the a37r2 is "1733530.1696830261".

[create a67r2]
a67r2 is a area. "An area (67) in r2".
the printed name of the a67r2 is "Room 2".
Understand "Area 67 in Room 2" as a67r2.
the parent of the a67r2 is r2.
the X of the a67r2 is "547975.0".
the Y of the a67r2 is "1733403.0".

[create a68r2]
a68r2 is a area. "An area (68) in r2".
the printed name of the a68r2 is "Room 2".
Understand "Area 68 in Room 2" as a68r2.
the parent of the a68r2 is r2.
the X of the a68r2 is "547931.8503889913".
the Y of the a68r2 is "1733461.5027281865".

[create a110r2]
a110r2 is a area. "An area (110) in r2".
the printed name of the a110r2 is "Room 2".
Understand "Area 110 in Room 2" as a110r2.
the parent of the a110r2 is r2.
the X of the a110r2 is "547785.0".
the Y of the a110r2 is "1733369.0".

[create a7r3]
a7r3 is a area. "An area (7) in r3".
the printed name of the a7r3 is "Room 3".
Understand "Area 7 in Room 3" as a7r3.
the parent of the a7r3 is r3.
the X of the a7r3 is "548744.9001965041".
the Y of the a7r3 is "1735193.946927291".

[create a8r3]
a8r3 is a area. "An area (8) in r3".
the printed name of the a8r3 is "Room 3".
Understand "Area 8 in Room 3" as a8r3.
the parent of the a8r3 is r3.
the X of the a8r3 is "548773.6187909034".
the Y of the a8r3 is "1735323.5461257007".

[create a15r3]
a15r3 is a area. "An area (15) in r3".
the printed name of the a15r3 is "Room 3".
Understand "Area 15 in Room 3" as a15r3.
the parent of the a15r3 is r3.
the X of the a15r3 is "549149.8606168446".
the Y of the a15r3 is "1735459.8602906289".

[create a16r3]
a16r3 is a area. "An area (16) in r3".
the printed name of the a16r3 is "Room 3".
Understand "Area 16 in Room 3" as a16r3.
the parent of the a16r3 is r3.
the X of the a16r3 is "548688.9999999998".
the Y of the a16r3 is "1735447.7422185438".

[create a55r3]
a55r3 is a area. "An area (55) in r3".
the printed name of the a55r3 is "Room 3".
Understand "Area 55 in Room 3" as a55r3.
the parent of the a55r3 is r3.
the X of the a55r3 is "549198.0".
the Y of the a55r3 is "1735514.0".

[create a56r3]
a56r3 is a area. "An area (56) in r3".
the printed name of the a56r3 is "Room 3".
Understand "Area 56 in Room 3" as a56r3.
the parent of the a56r3 is r3.
the X of the a56r3 is "549166.9123931623".
the Y of the a56r3 is "1735488.1662393163".

[create a71r3]
a71r3 is a area. "An area (71) in r3".
the printed name of the a71r3 is "Room 3".
Understand "Area 71 in Room 3" as a71r3.
the parent of the a71r3 is r3.
the X of the a71r3 is "549207.0".
the Y of the a71r3 is "1735424.0".

[create a72r3]
a72r3 is a area. "An area (72) in r3".
the printed name of the a72r3 is "Room 3".
Understand "Area 72 in Room 3" as a72r3.
the parent of the a72r3 is r3.
the X of the a72r3 is "549164.94".
the Y of the a72r3 is "1735442.5189999999".

[create a112r3]
a112r3 is a area. "An area (112) in r3".
the printed name of the a112r3 is "Room 3".
Understand "Area 112 in Room 3" as a112r3.
the parent of the a112r3 is r3.
the X of the a112r3 is "548654.0".
the Y of the a112r3 is "1735504.0".

[create a17r4]
a17r4 is a area. "An area (17) in r4".
the printed name of the a17r4 is "Room 4".
Understand "Area 17 in Room 4" as a17r4.
the parent of the a17r4 is r4.
the X of the a17r4 is "548918.5638533821".
the Y of the a17r4 is "1733728.3159999673".

[create a18r4]
a18r4 is a area. "An area (18) in r4".
the printed name of the a18r4 is "Room 4".
Understand "Area 18 in Room 4" as a18r4.
the parent of the a18r4 is r4.
the X of the a18r4 is "549102.4400057853".
the Y of the a18r4 is "1733949.016294183".

[create a19r4]
a19r4 is a area. "An area (19) in r4".
the printed name of the a19r4 is "Room 4".
Understand "Area 19 in Room 4" as a19r4.
the parent of the a19r4 is r4.
the X of the a19r4 is "549207.056747409".
the Y of the a19r4 is "1734243.5074638128".

[create a20r4]
a20r4 is a area. "An area (20) in r4".
the printed name of the a20r4 is "Room 4".
Understand "Area 20 in Room 4" as a20r4.
the parent of the a20r4 is r4.
the X of the a20r4 is "550076.9076679607".
the Y of the a20r4 is "1734256.9294065498".

[create a28r4]
a28r4 is a area. "An area (28) in r4".
the printed name of the a28r4 is "Room 4".
Understand "Area 28 in Room 4" as a28r4.
the parent of the a28r4 is r4.
the X of the a28r4 is "550313.4310969366".
the Y of the a28r4 is "1733567.4370029133".

[create a29r4]
a29r4 is a area. "An area (29) in r4".
the printed name of the a29r4 is "Room 4".
Understand "Area 29 in Room 4" as a29r4.
the parent of the a29r4 is r4.
the X of the a29r4 is "548902.7884243231".
the Y of the a29r4 is "1733612.7489356406".

[create a30r4]
a30r4 is a area. "An area (30) in r4".
the printed name of the a30r4 is "Room 4".
Understand "Area 30 in Room 4" as a30r4.
the parent of the a30r4 is r4.
the X of the a30r4 is "550679.7901234569".
the Y of the a30r4 is "1733443.2942386833".

[create a33r4]
a33r4 is a area. "An area (33) in r4".
the printed name of the a33r4 is "Room 4".
Understand "Area 33 in Room 4" as a33r4.
the parent of the a33r4 is r4.
the X of the a33r4 is "549231.5527700469".
the Y of the a33r4 is "1733447.12896628".

[create a34r4]
a34r4 is a area. "An area (34) in r4".
the printed name of the a34r4 is "Room 4".
Understand "Area 34 in Room 4" as a34r4.
the parent of the a34r4 is r4.
the X of the a34r4 is "548650.7616025948".
the Y of the a34r4 is "1733463.572044044".

[create a38r4]
a38r4 is a area. "An area (38) in r4".
the printed name of the a38r4 is "Room 4".
Understand "Area 38 in Room 4" as a38r4.
the parent of the a38r4 is r4.
the X of the a38r4 is "549325.6783433999".
the Y of the a38r4 is "1733444.643515631".

[create a40r4]
a40r4 is a area. "An area (40) in r4".
the printed name of the a40r4 is "Room 4".
Understand "Area 40 in Room 4" as a40r4.
the parent of the a40r4 is r4.
the X of the a40r4 is "549932.634469697".
the Y of the a40r4 is "1733444.0018939395".

[create a41r4]
a41r4 is a area. "An area (41) in r4".
the printed name of the a41r4 is "Room 4".
Understand "Area 41 in Room 4" as a41r4.
the parent of the a41r4 is r4.
the X of the a41r4 is "548468.6564395542".
the Y of the a41r4 is "1733440.4484777995".

[create a61r4]
a61r4 is a area. "An area (61) in r4".
the printed name of the a61r4 is "Room 4".
Understand "Area 61 in Room 4" as a61r4.
the parent of the a61r4 is r4.
the X of the a61r4 is "548613.0".
the Y of the a61r4 is "1733391.0".

[create a62r4]
a62r4 is a area. "An area (62) in r4".
the printed name of the a62r4 is "Room 4".
Understand "Area 62 in Room 4" as a62r4.
the parent of the a62r4 is r4.
the X of the a62r4 is "548645.0470213176".
the Y of the a62r4 is "1733398.6716145615".

[create a69r4]
a69r4 is a area. "An area (69) in r4".
the printed name of the a69r4 is "Room 4".
Understand "Area 69 in Room 4" as a69r4.
the parent of the a69r4 is r4.
the X of the a69r4 is "548983.6598631558".
the Y of the a69r4 is "1733946.9078496154".

[create a70r4]
a70r4 is a area. "An area (70) in r4".
the printed name of the a70r4 is "Room 4".
Understand "Area 70 in Room 4" as a70r4.
the parent of the a70r4 is r4.
the X of the a70r4 is "548996.520384776".
the Y of the a70r4 is "1733983.662370109".

[create a83r4]
a83r4 is a area. "An area (83) in r4".
the printed name of the a83r4 is "Room 4".
Understand "Area 83 in Room 4" as a83r4.
the parent of the a83r4 is r4.
the X of the a83r4 is "550736.0".
the Y of the a83r4 is "1733480.0".

[create a84r4]
a84r4 is a area. "An area (84) in r4".
the printed name of the a84r4 is "Room 4".
Understand "Area 84 in Room 4" as a84r4.
the parent of the a84r4 is r4.
the X of the a84r4 is "550696.3329344183".
the Y of the a84r4 is "1733461.5337482048".

[create a85r4]
a85r4 is a area. "An area (85) in r4".
the printed name of the a85r4 is "Room 4".
Understand "Area 85 in Room 4" as a85r4.
the parent of the a85r4 is r4.
the X of the a85r4 is "549381.0".
the Y of the a85r4 is "1733480.0".

[create a86r4]
a86r4 is a area. "An area (86) in r4".
the printed name of the a86r4 is "Room 4".
Understand "Area 86 in Room 4" as a86r4.
the parent of the a86r4 is r4.
the X of the a86r4 is "549340.7054722288".
the Y of the a86r4 is "1733461.5971994374".

[create a87r4]
a87r4 is a area. "An area (87) in r4".
the printed name of the a87r4 is "Room 4".
Understand "Area 87 in Room 4" as a87r4.
the parent of the a87r4 is r4.
the X of the a87r4 is "549876.0".
the Y of the a87r4 is "1733480.0".

[create a88r4]
a88r4 is a area. "An area (88) in r4".
the printed name of the a88r4 is "Room 4".
Understand "Area 88 in Room 4" as a88r4.
the parent of the a88r4 is r4.
the X of the a88r4 is "549917.3556858185".
the Y of the a88r4 is "1733461.7045075549".

[create a91r4]
a91r4 is a area. "An area (91) in r4".
the printed name of the a91r4 is "Room 4".
Understand "Area 91 in Room 4" as a91r4.
the parent of the a91r4 is r4.
the X of the a91r4 is "549265.5548513483".
the Y of the a91r4 is "1733395.9381769993".

[create a92r4]
a92r4 is a area. "An area (92) in r4".
the printed name of the a92r4 is "Room 4".
Understand "Area 92 in Room 4" as a92r4.
the parent of the a92r4 is r4.
the X of the a92r4 is "549233.0473953127".
the Y of the a92r4 is "1733397.0904402256".

[create a93r4]
a93r4 is a area. "An area (93) in r4".
the printed name of the a93r4 is "Room 4".
Understand "Area 93 in Room 4" as a93r4.
the parent of the a93r4 is r4.
the X of the a93r4 is "548424.0".
the Y of the a93r4 is "1733391.0".

[create a94r4]
a94r4 is a area. "An area (94) in r4".
the printed name of the a94r4 is "Room 4".
Understand "Area 94 in Room 4" as a94r4.
the parent of the a94r4 is r4.
the X of the a94r4 is "548454.2113743854".
the Y of the a94r4 is "1733416.9254353242".

[create a101r4]
a101r4 is a area. "An area (101) in r4".
the printed name of the a101r4 is "Room 4".
Understand "Area 101 in Room 4" as a101r4.
the parent of the a101r4 is r4.
the X of the a101r4 is "550727.0".
the Y of the a101r4 is "1733391.0".

[create a102r4]
a102r4 is a area. "An area (102) in r4".
the printed name of the a102r4 is "Room 4".
Understand "Area 102 in Room 4" as a102r4.
the parent of the a102r4 is r4.
the X of the a102r4 is "550695.7862838916".
the Y of the a102r4 is "1733416.633971292".

[create a103r4]
a103r4 is a area. "An area (103) in r4".
the printed name of the a103r4 is "Room 4".
Understand "Area 103 in Room 4" as a103r4.
the parent of the a103r4 is r4.
the X of the a103r4 is "549372.0".
the Y of the a103r4 is "1733391.0".

[create a104r4]
a104r4 is a area. "An area (104) in r4".
the printed name of the a104r4 is "Room 4".
Understand "Area 104 in Room 4" as a104r4.
the parent of the a104r4 is r4.
the X of the a104r4 is "549346.4938690206".
the Y of the a104r4 is "1733416.056799762".

[create a105r4]
a105r4 is a area. "An area (105) in r4".
the printed name of the a105r4 is "Room 4".
Understand "Area 105 in Room 4" as a105r4.
the parent of the a105r4 is r4.
the X of the a105r4 is "549885.0".
the Y of the a105r4 is "1733391.0".

[create a106r4]
a106r4 is a area. "An area (106) in r4".
the printed name of the a106r4 is "Room 4".
Understand "Area 106 in Room 4" as a106r4.
the parent of the a106r4 is r4.
the X of the a106r4 is "549916.2137161085".
the Y of the a106r4 is "1733416.633971292".

[create a107r4]
a107r4 is a area. "An area (107) in r4".
the printed name of the a107r4 is "Room 4".
Understand "Area 107 in Room 4" as a107r4.
the parent of the a107r4 is r4.
the X of the a107r4 is "548414.0".
the Y of the a107r4 is "1733480.0".

[create a108r4]
a108r4 is a area. "An area (108) in r4".
the printed name of the a108r4 is "Room 4".
Understand "Area 108 in Room 4" as a108r4.
the parent of the a108r4 is r4.
the X of the a108r4 is "548445.3505784653".
the Y of the a108r4 is "1733460.9916380297".

[create a4r5]
a4r5 is a area. "An area (4) in r5".
the printed name of the a4r5 is "Room 5".
Understand "Area 4 in Room 5" as a4r5.
the parent of the a4r5 is r5.
the X of the a4r5 is "549606.4509177421".
the Y of the a4r5 is "1735364.2052820404".

[create a5r5]
a5r5 is a area. "An area (5) in r5".
the printed name of the a5r5 is "Room 5".
Understand "Area 5 in Room 5" as a5r5.
the parent of the a5r5 is r5.
the X of the a5r5 is "549658.7557274036".
the Y of the a5r5 is "1735363.3385288815".

[create a6r5]
a6r5 is a area. "An area (6) in r5".
the printed name of the a6r5 is "Room 5".
Understand "Area 6 in Room 5" as a6r5.
the parent of the a6r5 is r5.
the X of the a6r5 is "549731.259315724".
the Y of the a6r5 is "1735479.5971839246".

[create a47r5]
a47r5 is a area. "An area (47) in r5".
the printed name of the a47r5 is "Room 5".
Understand "Area 47 in Room 5" as a47r5.
the parent of the a47r5 is r5.
the X of the a47r5 is "549492.0".
the Y of the a47r5 is "1735501.0".

[create a48r5]
a48r5 is a area. "An area (48) in r5".
the printed name of the a48r5 is "Room 5".
Understand "Area 48 in Room 5" as a48r5.
the parent of the a48r5 is r5.
the X of the a48r5 is "549532.1037433734".
the Y of the a48r5 is "1735446.9209844903".

[create a109r5]
a109r5 is a area. "An area (109) in r5".
the printed name of the a109r5 is "Room 5".
Understand "Area 109 in Room 5" as a109r5.
the parent of the a109r5 is r5.
the X of the a109r5 is "549759.0".
the Y of the a109r5 is "1735532.0".

[create d0]
d0 is a door. "A door between a26r0 and a10r1".
d0 is north of a26r0 and south of a10r1.

[create d1]
d1 is a door. "A door between a25r0 and a21r1".
d1 is north of a25r0 and south of a21r1.

[create d4]
d4 is a door. "A door between a27r2 and a7r3".
d4 is northeast of a27r2 and southwest of a7r3.

[create d6]
d6 is a door. "A door between a10r1 and a27r2".
d6 is southeast of a10r1 and northwest of a27r2.

[create d8]
d8 is a door. "A door between a7r3 and a17r4".
d8 is south of a7r3 and north of a17r4.

[create d10]
d10 is a door. "A door between a20r4 and a5r5".
d10 is north of a20r4 and south of a5r5.

[create d11]
d11 is a door. "A door between a19r4 and a4r5".
d11 is north of a19r4 and south of a4r5.

east of a0r1 is west of a1r1.

east of a2r1 is west of a3r1.

east of a4r5 is west of a5r5.

northeast of a5r5 is southwest of a6r5.

north of a7r3 is south of a8r3.

south of a9r1 is north of a10r1.

east of a11r1 is west of a12r1.

northeast of a9r1 is southwest of a13r1.

west of a11r1 is east of a14r1.

southeast of a1r1 is northwest of a9r1.

east of a8r3 is west of a15r3.

northwest of a8r3 is southeast of a16r3.

northeast of a17r4 is southwest of a18r4.

north of a18r4 is south of a19r4.

east of a19r4 is west of a20r4.

southwest of a0r1 is northeast of a21r1.

southeast of a11r1 is northwest of a21r1.

northeast of a22r0 is southwest of a23r0.

northeast of a24r0 is southwest of a25r0.

northwest of a23r0 is southeast of a25r0.

south of a20r4 is north of a28r4.

south of a17r4 is north of a29r4.

east of a28r4 is west of a30r4.

east of a24r0 is west of a31r0.

west of a24r0 is east of a32r0.

southeast of a29r4 is northwest of a33r4.

southwest of a29r4 is northeast of a34r4.

southeast of a26r0 is northwest of a35r0.

northeast of a36r2 is southwest of a37r2.

east of a23r0 is west of a26r0.

east of a33r4 is west of a38r4.

east of a35r0 is west of a39r0.

west of a28r4 is east of a40r4.

west of a34r4 is east of a41r4.

west of a22r0 is east of a42r0.

northwest of a43r0 is southeast of a44r0.

northeast of a45r1 is southwest of a46r1.

southeast of a47r5 is northwest of a48r5.

southeast of a49r1 is northwest of a50r1.

southeast of a51r1 is northwest of a52r1.

southwest of a53r1 is northeast of a54r1.

southwest of a55r3 is northeast of a56r3.

east of a57r1 is west of a58r1.

west of a59r1 is east of a60r1.

east of a61r4 is west of a62r4.

west of a63r0 is east of a64r0.

east of a65r0 is west of a66r0.

northwest of a67r2 is southeast of a68r2.

north of a69r4 is south of a70r4.

northwest of a71r3 is southeast of a72r3.

northeast of a73r1 is southwest of a74r1.

northwest of a75r1 is southeast of a76r1.

northeast of a77r0 is southwest of a78r0.

southeast of a79r0 is northwest of a80r0.

southwest of a81r0 is northeast of a82r0.

southwest of a83r4 is northeast of a84r4.

southwest of a85r4 is northeast of a86r4.

southeast of a87r4 is northwest of a88r4.

southeast of a89r0 is northwest of a90r0.

west of a91r4 is east of a92r4.

northeast of a93r4 is southwest of a94r4.

northeast of a95r0 is southwest of a96r0.

southwest of a97r0 is northeast of a98r0.

northwest of a99r0 is southeast of a100r0.

northwest of a101r4 is southeast of a102r4.

northwest of a103r4 is southeast of a104r4.

northeast of a105r4 is southwest of a106r4.

southeast of a107r4 is northwest of a108r4.

northeast of a6r5 is southwest of a109r5.

southwest of a36r2 is northeast of a110r2.

northeast of a13r1 is southwest of a111r1.

northwest of a16r3 is southeast of a112r3.

northwest of a14r1 is southeast of a52r1.

northeast of a12r1 is southwest of a54r1.

northeast of a15r3 is southwest of a56r3.

north of a3r1 is south of a58r1.

north of a1r1 is south of a60r1.

northwest of a0r1 is southeast of a3r1.

southeast of a15r3 is northwest of a72r3.

southwest of a2r1 is northeast of a74r1.

southeast of a12r1 is northwest of a76r1.

southwest of a14r1 is northeast of a46r1.

northwest of a4r5 is southeast of a48r5.

northwest of a2r1 is southeast of a50r1.

northwest of a42r0 is southeast of a90r0.

south of a33r4 is north of a92r4.

south of a34r4 is north of a62r4.

south of a35r0 is north of a64r0.

south of a22r0 is north of a66r0.

southeast of a37r2 is northwest of a68r2.

south of a27r2 is north of a37r2.

west of a18r4 is east of a70r4.

southwest of a40r4 is northeast of a106r4.

northwest of a41r4 is southeast of a108r4.

southwest of a42r0 is northeast of a78r0.

northwest of a32r0 is southeast of a80r0.

northeast of a31r0 is southwest of a82r0.

northeast of a30r4 is southwest of a84r4.

northeast of a38r4 is southwest of a86r4.

northwest of a40r4 is southeast of a88r4.

southeast of a39r0 is northwest of a44r0.

southwest of a41r4 is northeast of a94r4.

southwest of a32r0 is northeast of a96r0.

northeast of a39r0 is southwest of a98r0.

southeast of a31r0 is northwest of a100r0.

southeast of a30r4 is northwest of a102r4.

southeast of a38r4 is northwest of a104r4.

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
the player is in a106r4.

the orientation of the player is 6.
