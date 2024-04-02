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

[create a24r0]
a24r0 is a area. "An area (24) in r0".
the printed name of the a24r0 is "Room 0".
Understand "Area 24 in Room 0" as a24r0.
the parent of the a24r0 is r0.
the X of the a24r0 is "550890.0".
the Y of the a24r0 is "1256714.0".

[create a25r0]
a25r0 is a area. "An area (25) in r0".
the printed name of the a25r0 is "Room 0".
Understand "Area 25 in Room 0" as a25r0.
the parent of the a25r0 is r0.
the X of the a25r0 is "550859.1050372066".
the Y of the a25r0 is "1256687.7894485786".

[create a26r0]
a26r0 is a area. "An area (26) in r0".
the printed name of the a26r0 is "Room 0".
Understand "Area 26 in Room 0" as a26r0.
the parent of the a26r0 is r0.
the X of the a26r0 is "550125.0".
the Y of the a26r0 is "1256714.0".

[create a27r0]
a27r0 is a area. "An area (27) in r0".
the printed name of the a27r0 is "Room 0".
Understand "Area 27 in Room 0" as a27r0.
the parent of the a27r0 is r0.
the X of the a27r0 is "550156.0876068376".
the Y of the a27r0 is "1256688.166239316".

[create a42r0]
a42r0 is a area. "An area (42) in r0".
the printed name of the a42r0 is "Room 0".
Understand "Area 42 in Room 0" as a42r0.
the parent of the a42r0 is r0.
the X of the a42r0 is "550900.0".
the Y of the a42r0 is "1256624.0".

[create a43r0]
a43r0 is a area. "An area (43) in r0".
the printed name of the a43r0 is "Room 0".
Understand "Area 43 in Room 0" as a43r0.
the parent of the a43r0 is r0.
the X of the a43r0 is "550857.1798056157".
the Y of the a43r0 is "1256642.019978402".

[create a48r0]
a48r0 is a area. "An area (48) in r0".
the printed name of the a48r0 is "Room 0".
Understand "Area 48 in Room 0" as a48r0.
the parent of the a48r0 is r0.
the X of the a48r0 is "550116.0".
the Y of the a48r0 is "1256624.0".

[create a49r0]
a49r0 is a area. "An area (49) in r0".
the printed name of the a49r0 is "Room 0".
Understand "Area 49 in Room 0" as a49r0.
the parent of the a49r0 is r0.
the X of the a49r0 is "550143.2059471365".
the Y of the a49r0 is "1256644.0044052864".

[create a57r0]
a57r0 is a area. "An area (57) in r0".
the printed name of the a57r0 is "Room 0".
Understand "Area 57 in Room 0" as a57r0.
the parent of the a57r0 is r0.
the X of the a57r0 is "550170.5080993522".
the Y of the a57r0 is "1256664.2282217422".

[create a58r0]
a58r0 is a area. "An area (58) in r0".
the printed name of the a58r0 is "Room 0".
Understand "Area 58 in Room 0" as a58r0.
the parent of the a58r0 is r0.
the X of the a58r0 is "550842.1865386872".
the Y of the a58r0 is "1256659.6370670425".

[create a86r0]
a86r0 is a area. "An area (86) in r0".
the printed name of the a86r0 is "Room 0".
Understand "Area 86 in Room 0" as a86r0.
the parent of the a86r0 is r0.
the X of the a86r0 is "550524.8704484986".
the Y of the a86r0 is "1256576.4146870645".

[create a0r1]
a0r1 is a area. "An area (0) in r1".
the printed name of the a0r1 is "Room 1".
Understand "Area 0 in Room 1" as a0r1.
the parent of the a0r1 is r1.
the X of the a0r1 is "548421.0".
the Y of the a0r1 is "1254680.0".

[create a1r1]
a1r1 is a area. "An area (1) in r1".
the printed name of the a1r1 is "Room 1".
Understand "Area 1 in Room 1" as a1r1.
the parent of the a1r1 is r1.
the X of the a1r1 is "548452.350578465".
the Y of the a1r1 is "1254660.9916380297".

[create a2r1]
a2r1 is a area. "An area (2) in r1".
the printed name of the a2r1 is "Room 1".
Understand "Area 2 in Room 1" as a2r1.
the parent of the a2r1 is r1.
the X of the a2r1 is "550205.0".
the Y of the a2r1 is "1254563.0".

[create a3r1]
a3r1 is a area. "An area (3) in r1".
the printed name of the a3r1 is "Room 1".
Understand "Area 3 in Room 1" as a3r1.
the parent of the a3r1 is r1.
the X of the a3r1 is "550169.4166666666".
the Y of the a3r1 is "1254645.8310397554".

[create a8r1]
a8r1 is a area. "An area (8) in r1".
the printed name of the a8r1 is "Room 1".
Understand "Area 8 in Room 1" as a8r1.
the parent of the a8r1 is r1.
the X of the a8r1 is "550605.0".
the Y of the a8r1 is "1254999.0".

[create a9r1]
a9r1 is a area. "An area (9) in r1".
the printed name of the a9r1 is "Room 1".
Understand "Area 9 in Room 1" as a9r1.
the parent of the a9r1 is r1.
the X of the a9r1 is "550508.4006425084".
the Y of the a9r1 is "1255037.4374839372".

[create a20r1]
a20r1 is a area. "An area (20) in r1".
the printed name of the a20r1 is "Room 1".
Understand "Area 20 in Room 1" as a20r1.
the parent of the a20r1 is r1.
the X of the a20r1 is "548431.0".
the Y of the a20r1 is "1254591.0".

[create a21r1]
a21r1 is a area. "An area (21) in r1".
the printed name of the a21r1 is "Room 1".
Understand "Area 21 in Room 1" as a21r1.
the parent of the a21r1 is r1.
the X of the a21r1 is "548462.0234020579".
the Y of the a21r1 is "1254617.0166743887".

[create a65r1]
a65r1 is a area. "An area (65) in r1".
the printed name of the a65r1 is "Room 1".
Understand "Area 65 in Room 1" as a65r1.
the parent of the a65r1 is r1.
the X of the a65r1 is "549151.9490079246".
the Y of the a65r1 is "1254655.599259142".

[create a66r1]
a66r1 is a area. "An area (66) in r1".
the printed name of the a66r1 is "Room 1".
Understand "Area 66 in Room 1" as a66r1.
the parent of the a66r1 is r1.
the X of the a66r1 is "549153.5".
the Y of the a66r1 is "1254663.1540915086".

[create a67r1]
a67r1 is a area. "An area (67) in r1".
the printed name of the a67r1 is "Room 1".
Understand "Area 67 in Room 1" as a67r1.
the parent of the a67r1 is r1.
the X of the a67r1 is "550152.0218346368".
the Y of the a67r1 is "1254684.5039537789".

[create a70r1]
a70r1 is a area. "An area (70) in r1".
the printed name of the a70r1 is "Room 1".
Understand "Area 70 in Room 1" as a70r1.
the parent of the a70r1 is r1.
the X of the a70r1 is "549221.7529017443".
the Y of the a70r1 is "1254649.137096515".

[create a71r1]
a71r1 is a area. "An area (71) in r1".
the printed name of the a71r1 is "Room 1".
Understand "Area 71 in Room 1" as a71r1.
the parent of the a71r1 is r1.
the X of the a71r1 is "549303.4822688549".
the Y of the a71r1 is "1254758.4378163856".

[create a73r1]
a73r1 is a area. "An area (73) in r1".
the printed name of the a73r1 is "Room 1".
Understand "Area 73 in Room 1" as a73r1.
the parent of the a73r1 is r1.
the X of the a73r1 is "550483.3853736456".
the Y of the a73r1 is "1255047.6808904754".

[create a76r1]
a76r1 is a area. "An area (76) in r1".
the printed name of the a76r1 is "Room 1".
Understand "Area 76 in Room 1" as a76r1.
the parent of the a76r1 is r1.
the X of the a76r1 is "548475.913213344".
the Y of the a76r1 is "1254640.22214224".

[create a83r1]
a83r1 is a area. "An area (83) in r1".
the printed name of the a83r1 is "Room 1".
Understand "Area 83 in Room 1" as a83r1.
the parent of the a83r1 is r1.
the X of the a83r1 is "549140.6579974673".
the Y of the a83r1 is "1254936.5173569333".

[create a84r1]
a84r1 is a area. "An area (84) in r1".
the printed name of the a84r1 is "Room 1".
Understand "Area 84 in Room 1" as a84r1.
the parent of the a84r1 is r1.
the X of the a84r1 is "549222.9008682454".
the Y of the a84r1 is "1254828.7287836578".

[create a85r1]
a85r1 is a area. "An area (85) in r1".
the printed name of the a85r1 is "Room 1".
Understand "Area 85 in Room 1" as a85r1.
the parent of the a85r1 is r1.
the X of the a85r1 is "548971.1299147431".
the Y of the a85r1 is "1255099.0124721108".

[create a87r1]
a87r1 is a area. "An area (87) in r1".
the printed name of the a87r1 is "Room 1".
Understand "Area 87 in Room 1" as a87r1.
the parent of the a87r1 is r1.
the X of the a87r1 is "549348.677295557".
the Y of the a87r1 is "1254729.8440448157".

[create a88r1]
a88r1 is a area. "An area (88) in r1".
the printed name of the a88r1 is "Room 1".
Understand "Area 88 in Room 1" as a88r1.
the parent of the a88r1 is r1.
the X of the a88r1 is "548861.0668298497".
the Y of the a88r1 is "1254765.8958366967".

[create a89r1]
a89r1 is a area. "An area (89) in r1".
the printed name of the a89r1 is "Room 1".
Understand "Area 89 in Room 1" as a89r1.
the parent of the a89r1 is r1.
the X of the a89r1 is "549268.656363806".
the Y of the a89r1 is "1254613.8960687274".

[create a28r2]
a28r2 is a area. "An area (28) in r2".
the printed name of the a28r2 is "Room 2".
Understand "Area 28 in Room 2" as a28r2.
the parent of the a28r2 is r2.
the X of the a28r2 is "549499.0".
the Y of the a28r2 is "1256714.0".

[create a29r2]
a29r2 is a area. "An area (29) in r2".
the printed name of the a29r2 is "Room 2".
Understand "Area 29 in Room 2" as a29r2.
the parent of the a29r2 is r2.
the X of the a29r2 is "549467.9123931624".
the Y of the a29r2 is "1256688.1662393163".

[create a32r2]
a32r2 is a area. "An area (32) in r2".
the printed name of the a32r2 is "Room 2".
Understand "Area 32 in Room 2" as a32r2.
the parent of the a32r2 is r2.
the X of the a32r2 is "548688.0".
the Y of the a32r2 is "1256714.0".

[create a33r2]
a33r2 is a area. "An area (33) in r2".
the printed name of the a33r2 is "Room 2".
Understand "Area 33 in Room 2" as a33r2.
the parent of the a33r2 is r2.
the X of the a33r2 is "548719.8264604812".
the Y of the a33r2 is "1256705.767353952".

[create a34r2]
a34r2 is a area. "An area (34) in r2".
the printed name of the a34r2 is "Room 2".
Understand "Area 34 in Room 2" as a34r2.
the parent of the a34r2 is r2.
the X of the a34r2 is "549206.0".
the Y of the a34r2 is "1256704.0".

[create a35r2]
a35r2 is a area. "An area (35) in r2".
the printed name of the a35r2 is "Room 2".
Understand "Area 35 in Room 2" as a35r2.
the parent of the a35r2 is r2.
the X of the a35r2 is "549173.005982906".
the Y of the a35r2 is "1256705.7505982905".

[create a44r2]
a44r2 is a area. "An area (44) in r2".
the printed name of the a44r2 is "Room 2".
Understand "Area 44 in Room 2" as a44r2.
the parent of the a44r2 is r2.
the X of the a44r2 is "549508.0".
the Y of the a44r2 is "1256624.0".

[create a45r2]
a45r2 is a area. "An area (45) in r2".
the printed name of the a45r2 is "Room 2".
Understand "Area 45 in Room 2" as a45r2.
the parent of the a45r2 is r2.
the X of the a45r2 is "549465.94".
the Y of the a45r2 is "1256642.519".

[create a46r2]
a46r2 is a area. "An area (46) in r2".
the printed name of the a46r2 is "Room 2".
Understand "Area 46 in Room 2" as a46r2.
the parent of the a46r2 is r2.
the X of the a46r2 is "548687.0".
the Y of the a46r2 is "1256704.0".

[create a47r2]
a47r2 is a area. "An area (47) in r2".
the printed name of the a47r2 is "Room 2".
Understand "Area 47 in Room 2" as a47r2.
the parent of the a47r2 is r2.
the X of the a47r2 is "548674.0000000005".
the Y of the a47r2 is "1256656.6174944814".

[create a54r2]
a54r2 is a area. "An area (54) in r2".
the printed name of the a54r2 is "Room 2".
Understand "Area 54 in Room 2" as a54r2.
the parent of the a54r2 is r2.
the X of the a54r2 is "549162.2962987978".
the Y of the a54r2 is "1256600.260209825".

[create a55r2]
a55r2 is a area. "An area (55) in r2".
the printed name of the a55r2 is "Room 2".
Understand "Area 55 in Room 2" as a55r2.
the parent of the a55r2 is r2.
the X of the a55r2 is "549450.8606168446".
the Y of the a55r2 is "1256659.8602906289".

[create a56r2]
a56r2 is a area. "An area (56) in r2".
the printed name of the a56r2 is "Room 2".
Understand "Area 56 in Room 2" as a56r2.
the parent of the a56r2 is r2.
the X of the a56r2 is "548729.1291171542".
the Y of the a56r2 is "1256614.6013185487".

[create a80r2]
a80r2 is a area. "An area (80) in r2".
the printed name of the a80r2 is "Room 2".
Understand "Area 80 in Room 2" as a80r2.
the parent of the a80r2 is r2.
the X of the a80r2 is "548922.6453128336".
the Y of the a80r2 is "1256388.799487508".

[create a81r2]
a81r2 is a area. "An area (81) in r2".
the printed name of the a81r2 is "Room 2".
Understand "Area 81 in Room 2" as a81r2.
the parent of the a81r2 is r2.
the X of the a81r2 is "548850.2513327135".
the Y of the a81r2 is "1256423.0992885348".

[create a6r3]
a6r3 is a area. "An area (6) in r3".
the printed name of the a6r3 is "Room 3".
Understand "Area 6 in Room 3" as a6r3.
the parent of the a6r3 is r3.
the X of the a6r3 is "547982.0".
the Y of the a6r3 is "1254603.0".

[create a7r3]
a7r3 is a area. "An area (7) in r3".
the printed name of the a7r3 is "Room 3".
Understand "Area 7 in Room 3" as a7r3.
the parent of the a7r3 is r3.
the X of the a7r3 is "547938.8503889912".
the Y of the a7r3 is "1254661.502728187".

[create a50r3]
a50r3 is a area. "An area (50) in r3".
the printed name of the a50r3 is "Room 3".
Understand "Area 50 in Room 3" as a50r3.
the parent of the a50r3 is r3.
the X of the a50r3 is "547792.0".
the Y of the a50r3 is "1254569.0".

[create a51r3]
a51r3 is a area. "An area (51) in r3".
the printed name of the a51r3 is "Room 3".
Understand "Area 51 in Room 3" as a51r3.
the parent of the a51r3 is r3.
the X of the a51r3 is "547815.5661679944".
the Y of the a51r3 is "1254619.9341592465".

[create a69r3]
a69r3 is a area. "An area (69) in r3".
the printed name of the a69r3 is "Room 3".
Understand "Area 69 in Room 3" as a69r3.
the parent of the a69r3 is r3.
the X of the a69r3 is "547878.0091290362".
the Y of the a69r3 is "1254730.1696830264".

[create a72r3]
a72r3 is a area. "An area (72) in r3".
the printed name of the a72r3 is "Room 3".
Understand "Area 72 in Room 3" as a72r3.
the parent of the a72r3 is r3.
the X of the a72r3 is "547880.7606867261".
the Y of the a72r3 is "1254749.5866818966".

[create a30r4]
a30r4 is a area. "An area (30) in r4".
the printed name of the a30r4 is "Room 4".
Understand "Area 30 in Room 4" as a30r4.
the parent of the a30r4 is r4.
the X of the a30r4 is "546662.8663393344".
the Y of the a30r4 is "1256702.6633933443".

[create a31r4]
a31r4 is a area. "An area (31) in r4".
the printed name of the a31r4 is "Room 4".
Understand "Area 31 in Room 4" as a31r4.
the parent of the a31r4 is r4.
the X of the a31r4 is "546695.9594024089".
the Y of the a31r4 is "1256705.0790733977".

[create a40r4]
a40r4 is a area. "An area (40) in r4".
the printed name of the a40r4 is "Room 4".
Understand "Area 40 in Room 4" as a40r4.
the parent of the a40r4 is r4.
the X of the a40r4 is "545147.0".
the Y of the a40r4 is "1255927.0".

[create a41r4]
a41r4 is a area. "An area (41) in r4".
the printed name of the a41r4 is "Room 4".
Understand "Area 41 in Room 4" as a41r4.
the parent of the a41r4 is r4.
the X of the a41r4 is "545296.0294117647".
the Y of the a41r4 is "1255889.375".

[create a52r4]
a52r4 is a area. "An area (52) in r4".
the printed name of the a52r4 is "Room 4".
Understand "Area 52 in Room 4" as a52r4.
the parent of the a52r4 is r4.
the X of the a52r4 is "547199.0".
the Y of the a52r4 is "1256704.0".

[create a53r4]
a53r4 is a area. "An area (53) in r4".
the printed name of the a53r4 is "Room 4".
Understand "Area 53 in Room 4" as a53r4.
the parent of the a53r4 is r4.
the X of the a53r4 is "547166.8333333334".
the Y of the a53r4 is "1256648.8852465046".

[create a59r4]
a59r4 is a area. "An area (59) in r4".
the printed name of the a59r4 is "Room 4".
Understand "Area 59 in Room 4" as a59r4.
the parent of the a59r4 is r4.
the X of the a59r4 is "546702.6697319963".
the Y of the a59r4 is "1256649.9089116955".

[create a64r4]
a64r4 is a area. "An area (64) in r4".
the printed name of the a64r4 is "Room 4".
Understand "Area 64 in Room 4" as a64r4.
the parent of the a64r4 is r4.
the X of the a64r4 is "545306.9032078378".
the Y of the a64r4 is "1255886.8473628089".

[create a78r4]
a78r4 is a area. "An area (78) in r4".
the printed name of the a78r4 is "Room 4".
Understand "Area 78 in Room 4" as a78r4.
the parent of the a78r4 is r4.
the X of the a78r4 is "547094.020640536".
the Y of the a78r4 is "1256418.45585486".

[create a79r4]
a79r4 is a area. "An area (79) in r4".
the printed name of the a79r4 is "Room 4".
Understand "Area 79 in Room 4" as a79r4.
the parent of the a79r4 is r4.
the X of the a79r4 is "547064.5590174152".
the Y of the a79r4 is "1256503.2599186078".

[create a4r5]
a4r5 is a area. "An area (4) in r5".
the printed name of the a4r5 is "Room 5".
Understand "Area 4 in Room 5" as a4r5.
the parent of the a4r5 is r5.
the X of the a4r5 is "545737.0".
the Y of the a4r5 is "1254563.0".

[create a5r5]
a5r5 is a area. "An area (5) in r5".
the printed name of the a5r5 is "Room 5".
Understand "Area 5 in Room 5" as a5r5.
the parent of the a5r5 is r5.
the X of the a5r5 is "545773.0".
the Y of the a5r5 is "1254655.3867768592".

[create a10r5]
a10r5 is a area. "An area (10) in r5".
the printed name of the a10r5 is "Room 5".
Understand "Area 10 in Room 5" as a10r5.
the parent of the a10r5 is r5.
the X of the a10r5 is "545147.0".
the Y of the a10r5 is "1255113.0".

[create a11r5]
a11r5 is a area. "An area (11) in r5".
the printed name of the a11r5 is "Room 5".
Understand "Area 11 in Room 5" as a11r5.
the parent of the a11r5 is r5.
the X of the a11r5 is "545231.2818622056".
the Y of the a11r5 is "1255149.642857143".

[create a12r5]
a12r5 is a area. "An area (12) in r5".
the printed name of the a12r5 is "Room 5".
Understand "Area 12 in Room 5" as a12r5.
the parent of the a12r5 is r5.
the X of the a12r5 is "547224.0".
the Y of the a12r5 is "1255445.0".

[create a13r5]
a13r5 is a area. "An area (13) in r5".
the printed name of the a13r5 is "Room 5".
Understand "Area 13 in Room 5" as a13r5.
the parent of the a13r5 is r5.
the X of the a13r5 is "547242.2704191617".
the Y of the a13r5 is "1255473.6734131735".

[create a14r5]
a14r5 is a area. "An area (14) in r5".
the printed name of the a14r5 is "Room 5".
Understand "Area 14 in Room 5" as a14r5.
the parent of the a14r5 is r5.
the X of the a14r5 is "546524.0".
the Y of the a14r5 is "1255454.0".

[create a15r5]
a15r5 is a area. "An area (15) in r5".
the printed name of the a15r5 is "Room 5".
Understand "Area 15 in Room 5" as a15r5.
the parent of the a15r5 is r5.
the X of the a15r5 is "546566.2240682532".
the Y of the a15r5 is "1255472.207678491".

[create a16r5]
a16r5 is a area. "An area (16) in r5".
the printed name of the a16r5 is "Room 5".
Understand "Area 16 in Room 5" as a16r5.
the parent of the a16r5 is r5.
the X of the a16r5 is "547261.0".
the Y of the a16r5 is "1255543.0".

[create a17r5]
a17r5 is a area. "An area (17) in r5".
the printed name of the a17r5 is "Room 5".
Understand "Area 17 in Room 5" as a17r5.
the parent of the a17r5 is r5.
the X of the a17r5 is "547234.3618398677".
the Y of the a17r5 is "1255517.82872538".

[create a18r5]
a18r5 is a area. "An area (18) in r5".
the printed name of the a18r5 is "Room 5".
Understand "Area 18 in Room 5" as a18r5.
the parent of the a18r5 is r5.
the X of the a18r5 is "547380.0".
the Y of the a18r5 is "1254591.0".

[create a19r5]
a19r5 is a area. "An area (19) in r5".
the printed name of the a19r5 is "Room 5".
Understand "Area 19 in Room 5" as a19r5.
the parent of the a19r5 is r5.
the X of the a19r5 is "547348.9579422104".
the Y of the a19r5 is "1254616.6166125853".

[create a22r5]
a22r5 is a area. "An area (22) in r5".
the printed name of the a22r5 is "Room 5".
Understand "Area 22 in Room 5" as a22r5.
the parent of the a22r5 is r5.
the X of the a22r5 is "547389.0".
the Y of the a22r5 is "1254680.0".

[create a23r5]
a23r5 is a area. "An area (23) in r5".
the printed name of the a23r5 is "Room 5".
Understand "Area 23 in Room 5" as a23r5.
the parent of the a23r5 is r5.
the X of the a23r5 is "547361.849158728".
the Y of the a23r5 is "1254660.2680626006".

[create a36r5]
a36r5 is a area. "An area (36) in r5".
the printed name of the a36r5 is "Room 5".
Understand "Area 36 in Room 5" as a36r5.
the parent of the a36r5 is r5.
the X of the a36r5 is "547107.0".
the Y of the a36r5 is "1255383.0".

[create a37r5]
a37r5 is a area. "An area (37) in r5".
the printed name of the a37r5 is "Room 5".
Understand "Area 37 in Room 5" as a37r5.
the parent of the a37r5 is r5.
the X of the a37r5 is "547047.1666666665".
the Y of the a37r5 is "1255445.8333333333".

[create a38r5]
a38r5 is a area. "An area (38) in r5".
the printed name of the a38r5 is "Room 5".
Understand "Area 38 in Room 5" as a38r5.
the parent of the a38r5 is r5.
the X of the a38r5 is "546533.0".
the Y of the a38r5 is "1255543.0".

[create a39r5]
a39r5 is a area. "An area (39) in r5".
the printed name of the a39r5 is "Room 5".
Understand "Area 39 in Room 5" as a39r5.
the parent of the a39r5 is r5.
the X of the a39r5 is "546564.0420577896".
the Y of the a39r5 is "1255517.3833874147".

[create a60r5]
a60r5 is a area. "An area (60) in r5".
the printed name of the a60r5 is "Room 5".
Understand "Area 60 in Room 5" as a60r5.
the parent of the a60r5 is r5.
the X of the a60r5 is "547217.1316267735".
the Y of the a60r5 is "1255493.1735979232".

[create a61r5]
a61r5 is a area. "An area (61) in r5".
the printed name of the a61r5 is "Room 5".
Understand "Area 61 in Room 5" as a61r5.
the parent of the a61r5 is r5.
the X of the a61r5 is "546581.0200662521".
the Y of the a61r5 is "1255489.2230761899".

[create a62r5]
a62r5 is a area. "An area (62) in r5".
the printed name of the a62r5 is "Room 5".
Understand "Area 62 in Room 5" as a62r5.
the parent of the a62r5 is r5.
the X of the a62r5 is "547034.9393142136".
the Y of the a62r5 is "1255453.7451496266".

[create a63r5]
a63r5 is a area. "An area (63) in r5".
the printed name of the a63r5 is "Room 5".
Understand "Area 63 in Room 5" as a63r5.
the parent of the a63r5 is r5.
the X of the a63r5 is "547007.3663366542".
the Y of the a63r5 is "1255451.485902589".

[create a68r5]
a68r5 is a area. "An area (68) in r5".
the printed name of the a68r5 is "Room 5".
Understand "Area 68 in Room 5" as a68r5.
the parent of the a68r5 is r5.
the X of the a68r5 is "545790.3005528573".
the Y of the a68r5 is "1254698.8583478576".

[create a74r5]
a74r5 is a area. "An area (74) in r5".
the printed name of the a74r5 is "Room 5".
Understand "Area 74 in Room 5" as a74r5.
the parent of the a74r5 is r5.
the X of the a74r5 is "545327.6742984809".
the Y of the a74r5 is "1255191.5791008822".

[create a75r5]
a75r5 is a area. "An area (75) in r5".
the printed name of the a75r5 is "Room 5".
Understand "Area 75 in Room 5" as a75r5.
the parent of the a75r5 is r5.
the X of the a75r5 is "547334.8228866081".
the Y of the a75r5 is "1254640.061504006".

[create a77r5]
a77r5 is a area. "An area (77) in r5".
the printed name of the a77r5 is "Room 5".
Understand "Area 77 in Room 5" as a77r5.
the parent of the a77r5 is r5.
the X of the a77r5 is "546935.1172780078".
the Y of the a77r5 is "1255368.8996662495".

[create a82r5]
a82r5 is a area. "An area (82) in r5".
the printed name of the a82r5 is "Room 5".
Understand "Area 82 in Room 5" as a82r5.
the parent of the a82r5 is r5.
the X of the a82r5 is "546946.7781649684".
the Y of the a82r5 is "1254761.713512533".

[create d0]
d0 is a door. "A door between a81r2 and a72r3".
d0 is southwest of a81r2 and northeast of a72r3.

[create d2]
d2 is a door. "A door between a72r3 and a78r4".
d2 is northwest of a72r3 and southeast of a78r4.

[create d4]
d4 is a door. "A door between a86r0 and a73r1".
d4 is south of a86r0 and north of a73r1.

[create d6]
d6 is a door. "A door between a85r1 and a80r2".
d6 is north of a85r1 and south of a80r2.

[create d8]
d8 is a door. "A door between a78r4 and a63r5".
d8 is south of a78r4 and north of a63r5.

[create d9]
d9 is a door. "A door between a64r4 and a74r5".
d9 is south of a64r4 and north of a74r5.

southeast of a0r1 is northwest of a1r1.

northwest of a2r1 is southeast of a3r1.

north of a4r5 is south of a5r5.

northwest of a6r3 is southeast of a7r3.

west of a8r1 is east of a9r1.

northeast of a10r5 is southwest of a11r5.

northeast of a12r5 is southwest of a13r5.

northeast of a14r5 is southwest of a15r5.

southwest of a16r5 is northeast of a17r5.

northwest of a18r5 is southeast of a19r5.

northeast of a20r1 is southwest of a21r1.

southwest of a22r5 is northeast of a23r5.

southwest of a24r0 is northeast of a25r0.

southeast of a26r0 is northwest of a27r0.

southwest of a28r2 is northeast of a29r2.

east of a30r4 is west of a31r4.

east of a32r2 is west of a33r2.

west of a34r2 is east of a35r2.

northwest of a36r5 is southeast of a37r5.

southeast of a38r5 is northwest of a39r5.

east of a40r4 is west of a41r4.

northwest of a42r0 is southeast of a43r0.

northwest of a44r2 is southeast of a45r2.

south of a46r2 is north of a47r2.

northeast of a48r0 is southwest of a49r0.

northeast of a50r3 is southwest of a51r3.

southwest of a52r4 is northeast of a53r4.

south of a35r2 is north of a54r2.

northwest of a45r2 is southeast of a55r2.

southeast of a47r2 is northwest of a56r2.

northeast of a49r0 is southwest of a57r0.

southwest of a25r0 is northeast of a58r0.

southeast of a27r0 is northwest of a57r0.

southwest of a29r2 is northeast of a55r2.

south of a31r4 is north of a59r4.

south of a33r2 is north of a56r2.

northwest of a13r5 is southeast of a60r5.

northeast of a15r5 is southwest of a61r5.

southwest of a17r5 is northeast of a60r5.

northwest of a37r5 is southeast of a62r5.

southeast of a39r5 is northwest of a61r5.

west of a62r5 is east of a63r5.

east of a41r4 is west of a64r4.

northwest of a43r0 is southeast of a58r0.

north of a65r1 is south of a66r1.

northwest of a3r1 is southeast of a67r1.

north of a5r5 is south of a68r5.

northwest of a7r3 is southeast of a69r3.

northeast of a70r1 is southwest of a71r1.

north of a69r3 is south of a72r3.

west of a9r1 is east of a73r1.

northeast of a11r5 is southwest of a74r5.

northwest of a19r5 is southeast of a75r5.

northeast of a21r1 is southwest of a76r1.

southwest of a23r5 is northeast of a75r5.

southeast of a1r1 is northwest of a76r1.

southwest of a63r5 is northeast of a77r5.

southwest of a59r4 is northeast of a64r4.

north of a78r4 is south of a79r4.

southwest of a53r4 is northeast of a79r4.

southwest of a54r2 is northeast of a80r2.

northwest of a80r2 is southeast of a81r2.

east of a59r4 is west of a79r4.

east of a54r2 is west of a55r2.

south of a77r5 is north of a82r5.

southeast of a83r1 is northwest of a84r1.

northwest of a83r1 is southeast of a85r1.

east of a61r5 is west of a77r5.

west of a60r5 is east of a62r5.

northeast of a51r3 is southwest of a69r3.

northeast of a67r1 is southwest of a73r1.

northwest of a68r5 is southeast of a74r5.

southeast of a71r1 is northwest of a87r1.

northwest of a71r1 is southeast of a84r1.

south of a85r1 is north of a88r1.

southeast of a70r1 is northwest of a89r1.

west of a75r5 is east of a82r5.

east of a76r1 is west of a88r1.

east of a65r1 is west of a70r1.

west of a65r1 is east of a88r1.

east of a68r5 is west of a82r5.

west of a67r1 is east of a87r1.

southeast of a56r2 is northwest of a81r2.

east of a57r0 is west of a86r0.

west of a58r0 is east of a86r0.

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
the player is in a87r1.

the orientation of the player is 6.
