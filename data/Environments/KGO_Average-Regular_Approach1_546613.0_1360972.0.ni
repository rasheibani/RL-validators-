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

[create a18r0]
a18r0 is a area. "An area (18) in r0".
the printed name of the a18r0 is "Room 0".
Understand "Area 18 in Room 0" as a18r0.
the parent of the a18r0 is r0.
the X of the a18r0 is "545545.6602166065".
the Y of the a18r0 is "1362015.0".

[create a35r0]
a35r0 is a area. "An area (35) in r0".
the printed name of the a35r0 is "Room 0".
Understand "Area 35 in Room 0" as a35r0.
the parent of the a35r0 is r0.
the X of the a35r0 is "545525.0".
the Y of the a35r0 is "1361168.5778301887".

[create a55r0]
a55r0 is a area. "An area (55) in r0".
the printed name of the a55r0 is "Room 0".
Understand "Area 55 in Room 0" as a55r0.
the parent of the a55r0 is r0.
the X of the a55r0 is "545949.0".
the Y of the a55r0 is "1361080.0".

[create a56r0]
a56r0 is a area. "An area (56) in r0".
the printed name of the a56r0 is "Room 0".
Understand "Area 56 in Room 0" as a56r0.
the parent of the a56r0 is r0.
the X of the a56r0 is "545906.1485252036".
the Y of the a56r0 is "1361061.855767114".

[create a71r0]
a71r0 is a area. "An area (71) in r0".
the printed name of the a71r0 is "Room 0".
Understand "Area 71 in Room 0" as a71r0.
the parent of the a71r0 is r0.
the X of the a71r0 is "545098.0".
the Y of the a71r0 is "1360991.0".

[create a72r0]
a72r0 is a area. "An area (72) in r0".
the printed name of the a72r0 is "Room 0".
Understand "Area 72 in Room 0" as a72r0.
the parent of the a72r0 is r0.
the X of the a72r0 is "545129.2137161085".
the Y of the a72r0 is "1361016.6339712918".

[create a75r0]
a75r0 is a area. "An area (75) in r0".
the printed name of the a75r0 is "Room 0".
Understand "Area 75 in Room 0" as a75r0.
the parent of the a75r0 is r0.
the X of the a75r0 is "545089.0".
the Y of the a75r0 is "1361080.0".

[create a76r0]
a76r0 is a area. "An area (76) in r0".
the printed name of the a76r0 is "Room 0".
Understand "Area 76 in Room 0" as a76r0.
the parent of the a76r0 is r0.
the X of the a76r0 is "545120.0059757694".
the Y of the a76r0 is "1361060.657907662".

[create a77r0]
a77r0 is a area. "An area (77) in r0".
the printed name of the a77r0 is "Room 0".
Understand "Area 77 in Room 0" as a77r0.
the parent of the a77r0 is r0.
the X of the a77r0 is "545940.0".
the Y of the a77r0 is "1360991.0".

[create a78r0]
a78r0 is a area. "An area (78) in r0".
the printed name of the a78r0 is "Room 0".
Understand "Area 78 in Room 0" as a78r0.
the parent of the a78r0 is r0.
the X of the a78r0 is "545908.7862838916".
the Y of the a78r0 is "1361016.6339712918".

[create a94r0]
a94r0 is a area. "An area (94) in r0".
the printed name of the a94r0 is "Room 0".
Understand "Area 94 in Room 0" as a94r0.
the parent of the a94r0 is r0.
the X of the a94r0 is "545143.389199255".
the Y of the a94r0 is "1361040.2597765361".

[create a95r0]
a95r0 is a area. "An area (95) in r0".
the printed name of the a95r0 is "Room 0".
Understand "Area 95 in Room 0" as a95r0.
the parent of the a95r0 is r0.
the X of the a95r0 is "545891.7817693062".
the Y of the a95r0 is "1361044.9748289345".

[create a8r1]
a8r1 is a area. "An area (8) in r1".
the printed name of the a8r1 is "Room 1".
Understand "Area 8 in Room 1" as a8r1.
the parent of the a8r1 is r1.
the X of the a8r1 is "545168.0926719012".
the Y of the a8r1 is "1363059.8065726864".

[create a9r1]
a9r1 is a area. "An area (9) in r1".
the printed name of the a9r1 is "Room 1".
Understand "Area 9 in Room 1" as a9r1.
the parent of the a9r1 is r1.
the X of the a9r1 is "545525.0".
the Y of the a9r1 is "1362935.3372641508".

[create a10r1]
a10r1 is a area. "An area (10) in r1".
the printed name of the a10r1 is "Room 1".
Understand "Area 10 in Room 1" as a10r1.
the parent of the a10r1 is r1.
the X of the a10r1 is "545882.8606168448".
the Y of the a10r1 is "1363059.8602906289".

[create a11r1]
a11r1 is a area. "An area (11) in r1".
the printed name of the a11r1 is "Room 1".
Understand "Area 11 in Room 1" as a11r1.
the parent of the a11r1 is r1.
the X of the a11r1 is "546612.6739697324".
the Y of the a11r1 is "1362978.3395440248".

[create a12r1]
a12r1 is a area. "An area (12) in r1".
the printed name of the a12r1 is "Room 1".
Understand "Area 12 in Room 1" as a12r1.
the parent of the a12r1 is r1.
the X of the a12r1 is "546981.5734589623".
the Y of the a12r1 is "1363062.7036085443".

[create a19r1]
a19r1 is a area. "An area (19) in r1".
the printed name of the a19r1 is "Room 1".
Understand "Area 19 in Room 1" as a19r1.
the parent of the a19r1 is r1.
the X of the a19r1 is "545913.7347933757".
the Y of the a19r1 is "1362169.6178680474".

[create a36r1]
a36r1 is a area. "An area (36) in r1".
the printed name of the a36r1 is "Room 1".
Understand "Area 36 in Room 1" as a36r1.
the parent of the a36r1 is r1.
the X of the a36r1 is "546317.4139571368".
the Y of the a36r1 is "1363059.275809477".

[create a51r1]
a51r1 is a area. "An area (51) in r1".
the printed name of the a51r1 is "Room 1".
Understand "Area 51 in Room 1" as a51r1.
the parent of the a51r1 is r1.
the X of the a51r1 is "546260.0".
the Y of the a51r1 is "1363024.0".

[create a52r1]
a52r1 is a area. "An area (52) in r1".
the printed name of the a52r1 is "Room 1".
Understand "Area 52 in Room 1" as a52r1.
the parent of the a52r1 is r1.
the X of the a52r1 is "546303.1182354635".
the Y of the a52r1 is "1363042.4131764537".

[create a53r1]
a53r1 is a area. "An area (53) in r1".
the printed name of the a53r1 is "Room 1".
Understand "Area 53 in Room 1" as a53r1.
the parent of the a53r1 is r1.
the X of the a53r1 is "545111.0".
the Y of the a53r1 is "1363024.0".

[create a54r1]
a54r1 is a area. "An area (54) in r1".
the printed name of the a54r1 is "Room 1".
Understand "Area 54 in Room 1" as a54r1.
the parent of the a54r1 is r1.
the X of the a54r1 is "545153.06".
the Y of the a54r1 is "1363042.5189999999".

[create a79r1]
a79r1 is a area. "An area (79) in r1".
the printed name of the a79r1 is "Room 1".
Understand "Area 79 in Room 1" as a79r1.
the parent of the a79r1 is r1.
the X of the a79r1 is "545940.0".
the Y of the a79r1 is "1363024.0".

[create a80r1]
a80r1 is a area. "An area (80) in r1".
the printed name of the a80r1 is "Room 1".
Understand "Area 80 in Room 1" as a80r1.
the parent of the a80r1 is r1.
the X of the a80r1 is "545897.94".
the Y of the a80r1 is "1363042.519".

[create a81r1]
a81r1 is a area. "An area (81) in r1".
the printed name of the a81r1 is "Room 1".
Understand "Area 81 in Room 1" as a81r1.
the parent of the a81r1 is r1.
the X of the a81r1 is "547037.0".
the Y of the a81r1 is "1363024.0".

[create a82r1]
a82r1 is a area. "An area (82) in r1".
the printed name of the a82r1 is "Room 1".
Understand "Area 82 in Room 1" as a82r1.
the parent of the a82r1 is r1.
the X of the a82r1 is "547001.2162216022".
the Y of the a82r1 is "1363043.14662216".

[create a83r1]
a83r1 is a area. "An area (83) in r1".
the printed name of the a83r1 is "Room 1".
Understand "Area 83 in Room 1" as a83r1.
the parent of the a83r1 is r1.
the X of the a83r1 is "545120.0".
the Y of the a83r1 is "1363114.0".

[create a84r1]
a84r1 is a area. "An area (84) in r1".
the printed name of the a84r1 is "Room 1".
Understand "Area 84 in Room 1" as a84r1.
the parent of the a84r1 is r1.
the X of the a84r1 is "545150.9153780071".
the Y of the a84r1 is "1363088.183462199".

[create a85r1]
a85r1 is a area. "An area (85) in r1".
the printed name of the a85r1 is "Room 1".
Understand "Area 85 in Room 1" as a85r1.
the parent of the a85r1 is r1.
the X of the a85r1 is "546269.0".
the Y of the a85r1 is "1363114.0".

[create a86r1]
a86r1 is a area. "An area (86) in r1".
the printed name of the a86r1 is "Room 1".
Understand "Area 86 in Room 1" as a86r1.
the parent of the a86r1 is r1.
the X of the a86r1 is "546299.9153780069".
the Y of the a86r1 is "1363088.1834621993".

[create a87r1]
a87r1 is a area. "An area (87) in r1".
the printed name of the a87r1 is "Room 1".
Understand "Area 87 in Room 1" as a87r1.
the parent of the a87r1 is r1.
the X of the a87r1 is "545931.0".
the Y of the a87r1 is "1363114.0".

[create a88r1]
a88r1 is a area. "An area (88) in r1".
the printed name of the a88r1 is "Room 1".
Understand "Area 88 in Room 1" as a88r1.
the parent of the a88r1 is r1.
the X of the a88r1 is "545899.9123931623".
the Y of the a88r1 is "1363088.1662393163".

[create a89r1]
a89r1 is a area. "An area (89) in r1".
the printed name of the a89r1 is "Room 1".
Understand "Area 89 in Room 1" as a89r1.
the parent of the a89r1 is r1.
the X of the a89r1 is "547028.0".
the Y of the a89r1 is "1363114.0".

[create a90r1]
a90r1 is a area. "An area (90) in r1".
the printed name of the a90r1 is "Room 1".
Understand "Area 90 in Room 1" as a90r1.
the parent of the a90r1 is r1.
the X of the a90r1 is "546996.9123931623".
the Y of the a90r1 is "1363088.1662393163".

[create a13r2]
a13r2 is a area. "An area (13) in r2".
the printed name of the a13r2 is "Room 2".
Understand "Area 13 in Room 2" as a13r2.
the parent of the a13r2 is r2.
the X of the a13r2 is "548490.8477146518".
the Y of the a13r2 is "1361847.3338316958".

[create a14r2]
a14r2 is a area. "An area (14) in r2".
the printed name of the a14r2 is "Room 2".
Understand "Area 14 in Room 2" as a14r2.
the parent of the a14r2 is r2.
the X of the a14r2 is "548661.8618197278".
the Y of the a14r2 is "1361894.158588435".

[create a15r2]
a15r2 is a area. "An area (15) in r2".
the printed name of the a15r2 is "Room 2".
Understand "Area 15 in Room 2" as a15r2.
the parent of the a15r2 is r2.
the X of the a15r2 is "548280.576746529".
the Y of the a15r2 is "1361879.0083536007".

[create a16r2]
a16r2 is a area. "An area (16) in r2".
the printed name of the a16r2 is "Room 2".
Understand "Area 16 in Room 2" as a16r2.
the parent of the a16r2 is r2.
the X of the a16r2 is "548333.8502184764".
the Y of the a16r2 is "1361749.3841743374".

[create a21r2]
a21r2 is a area. "An area (21) in r2".
the printed name of the a21r2 is "Room 2".
Understand "Area 21 in Room 2" as a21r2.
the parent of the a21r2 is r2.
the X of the a21r2 is "546455.9253641139".
the Y of the a21r2 is "1361447.354039811".

[create a22r2]
a22r2 is a area. "An area (22) in r2".
the printed name of the a22r2 is "Room 2".
Understand "Area 22 in Room 2" as a22r2.
the parent of the a22r2 is r2.
the X of the a22r2 is "546502.4961703713".
the Y of the a22r2 is "1361384.5755879027".

[create a23r2]
a23r2 is a area. "An area (23) in r2".
the printed name of the a23r2 is "Room 2".
Understand "Area 23 in Room 2" as a23r2.
the parent of the a23r2 is r2.
the X of the a23r2 is "546773.6407481865".
the Y of the a23r2 is "1361591.6719493715".

[create a24r2]
a24r2 is a area. "An area (24) in r2".
the printed name of the a24r2 is "Room 2".
Understand "Area 24 in Room 2" as a24r2.
the parent of the a24r2 is r2.
the X of the a24r2 is "546796.7982876387".
the Y of the a24r2 is "1361559.798753249".

[create a25r2]
a25r2 is a area. "An area (25) in r2".
the printed name of the a25r2 is "Room 2".
Understand "Area 25 in Room 2" as a25r2.
the parent of the a25r2 is r2.
the X of the a25r2 is "548330.4679417008".
the Y of the a25r2 is "1361574.3869784428".

[create a26r2]
a26r2 is a area. "An area (26) in r2".
the printed name of the a26r2 is "Room 2".
Understand "Area 26 in Room 2" as a26r2.
the parent of the a26r2 is r2.
the X of the a26r2 is "550068.652692378".
the Y of the a26r2 is "1361719.6370413492".

[create a27r2]
a27r2 is a area. "An area (27) in r2".
the printed name of the a27r2 is "Room 2".
Understand "Area 27 in Room 2" as a27r2.
the parent of the a27r2 is r2.
the X of the a27r2 is "548027.8783746968".
the Y of the a27r2 is "1361890.210130901".

[create a28r2]
a28r2 is a area. "An area (28) in r2".
the printed name of the a28r2 is "Room 2".
Understand "Area 28 in Room 2" as a28r2.
the parent of the a28r2 is r2.
the X of the a28r2 is "547165.1327880265".
the Y of the a28r2 is "1360998.6453184825".

[create a29r2]
a29r2 is a area. "An area (29) in r2".
the printed name of the a29r2 is "Room 2".
Understand "Area 29 in Room 2" as a29r2.
the parent of the a29r2 is r2.
the X of the a29r2 is "547236.0885318213".
the Y of the a29r2 is "1361098.9005518877".

[create a30r2]
a30r2 is a area. "An area (30) in r2".
the printed name of the a30r2 is "Room 2".
Understand "Area 30 in Room 2" as a30r2.
the parent of the a30r2 is r2.
the X of the a30r2 is "548426.4479693415".
the Y of the a30r2 is "1361321.7627812584".

[create a31r2]
a31r2 is a area. "An area (31) in r2".
the printed name of the a31r2 is "Room 2".
Understand "Area 31 in Room 2" as a31r2.
the parent of the a31r2 is r2.
the X of the a31r2 is "548672.5721387332".
the Y of the a31r2 is "1361095.1713213234".

[create a32r2]
a32r2 is a area. "An area (32) in r2".
the printed name of the a32r2 is "Room 2".
Understand "Area 32 in Room 2" as a32r2.
the parent of the a32r2 is r2.
the X of the a32r2 is "547172.0268316669".
the Y of the a32r2 is "1361139.3053511411".

[create a33r2]
a33r2 is a area. "An area (33) in r2".
the printed name of the a33r2 is "Room 2".
Understand "Area 33 in Room 2" as a33r2.
the parent of the a33r2 is r2.
the X of the a33r2 is "547099.1344889463".
the Y of the a33r2 is "1361196.8381297828".

[create a34r2]
a34r2 is a area. "An area (34) in r2".
the printed name of the a34r2 is "Room 2".
Understand "Area 34 in Room 2" as a34r2.
the parent of the a34r2 is r2.
the X of the a34r2 is "546690.6494994545".
the Y of the a34r2 is "1361130.2617805514".

[create a37r2]
a37r2 is a area. "An area (37) in r2".
the printed name of the a37r2 is "Room 2".
Understand "Area 37 in Room 2" as a37r2.
the parent of the a37r2 is r2.
the X of the a37r2 is "548707.0".
the Y of the a37r2 is "1361943.0".

[create a38r2]
a38r2 is a area. "An area (38) in r2".
the printed name of the a38r2 is "Room 2".
Understand "Area 38 in Room 2" as a38r2.
the parent of the a38r2 is r2.
the X of the a38r2 is "548675.7862838914".
the Y of the a38r2 is "1361917.3660287082".

[create a39r2]
a39r2 is a area. "An area (39) in r2".
the printed name of the a39r2 is "Room 2".
Understand "Area 39 in Room 2" as a39r2.
the parent of the a39r2 is r2.
the X of the a39r2 is "547979.0".
the Y of the a39r2 is "1361943.0".

[create a40r2]
a40r2 is a area. "An area (40) in r2".
the printed name of the a40r2 is "Room 2".
Understand "Area 40 in Room 2" as a40r2.
the parent of the a40r2 is r2.
the X of the a40r2 is "548013.1175929825".
the Y of the a40r2 is "1361917.0723782377".

[create a57r2]
a57r2 is a area. "An area (57) in r2".
the printed name of the a57r2 is "Room 2".
Understand "Area 57 in Room 2" as a57r2.
the parent of the a57r2 is r2.
the X of the a57r2 is "548608.0".
the Y of the a57r2 is "1360951.0".

[create a58r2]
a58r2 is a area. "An area (58) in r2".
the printed name of the a58r2 is "Room 2".
Understand "Area 58 in Room 2" as a58r2.
the parent of the a58r2 is r2.
the X of the a58r2 is "548644.0".
the Y of the a58r2 is "1361024.605439442".

[create a59r2]
a59r2 is a area. "An area (59) in r2".
the printed name of the a59r2 is "Room 2".
Understand "Area 59 in Room 2" as a59r2.
the parent of the a59r2 is r2.
the X of the a59r2 is "546613.0".
the Y of the a59r2 is "1360972.0".

[create a60r2]
a60r2 is a area. "An area (60) in r2".
the printed name of the a60r2 is "Room 2".
Understand "Area 60 in Room 2" as a60r2.
the parent of the a60r2 is r2.
the X of the a60r2 is "546646.9784514521".
the Y of the a60r2 is "1361045.2960965103".

[create a61r2]
a61r2 is a area. "An area (61) in r2".
the printed name of the a61r2 is "Room 2".
Understand "Area 61 in Room 2" as a61r2.
the parent of the a61r2 is r2.
the X of the a61r2 is "548120.0".
the Y of the a61r2 is "1361495.0".

[create a62r2]
a62r2 is a area. "An area (62) in r2".
the printed name of the a62r2 is "Room 2".
Understand "Area 62 in Room 2" as a62r2.
the parent of the a62r2 is r2.
the X of the a62r2 is "548200.1674687729".
the Y of the a62r2 is "1361527.6591836736".

[create a63r2]
a63r2 is a area. "An area (63) in r2".
the printed name of the a63r2 is "Room 2".
Understand "Area 63 in Room 2" as a63r2.
the parent of the a63r2 is r2.
the X of the a63r2 is "550233.0".
the Y of the a63r2 is "1361664.0".

[create a64r2]
a64r2 is a area. "An area (64) in r2".
the printed name of the a64r2 is "Room 2".
Understand "Area 64 in Room 2" as a64r2.
the parent of the a64r2 is r2.
the X of the a64r2 is "550120.6812015502".
the Y of the a64r2 is "1361703.1666666667".

[create a65r2]
a65r2 is a area. "An area (65) in r2".
the printed name of the a65r2 is "Room 2".
Understand "Area 65 in Room 2" as a65r2.
the parent of the a65r2 is r2.
the X of the a65r2 is "548575.0".
the Y of the a65r2 is "1361817.0".

[create a66r2]
a66r2 is a area. "An area (66) in r2".
the printed name of the a66r2 is "Room 2".
Understand "Area 66 in Room 2" as a66r2.
the parent of the a66r2 is r2.
the X of the a66r2 is "548493.1666666665".
the Y of the a66r2 is "1361845.8333333337".

[create a67r2]
a67r2 is a area. "An area (67) in r2".
the printed name of the a67r2 is "Room 2".
Understand "Area 67 in Room 2" as a67r2.
the parent of the a67r2 is r2.
the X of the a67r2 is "548716.0".
the Y of the a67r2 is "1361854.0".

[create a68r2]
a68r2 is a area. "An area (68) in r2".
the printed name of the a68r2 is "Room 2".
Understand "Area 68 in Room 2" as a68r2.
the parent of the a68r2 is r2.
the X of the a68r2 is "548688.2704191618".
the Y of the a68r2 is "1361873.6734131738".

[create a69r2]
a69r2 is a area. "An area (69) in r2".
the printed name of the a69r2 is "Room 2".
Understand "Area 69 in Room 2" as a69r2.
the parent of the a69r2 is r2.
the X of the a69r2 is "547970.0".
the Y of the a69r2 is "1361854.0".

[create a70r2]
a70r2 is a area. "An area (70) in r2".
the printed name of the a70r2 is "Room 2".
Understand "Area 70 in Room 2" as a70r2.
the parent of the a70r2 is r2.
the X of the a70r2 is "548012.2240682533".
the Y of the a70r2 is "1361872.207678491".

[create a73r2]
a73r2 is a area. "An area (73) in r2".
the printed name of the a73r2 is "Room 2".
Understand "Area 73 in Room 2" as a73r2.
the parent of the a73r2 is r2.
the X of the a73r2 is "547128.3228332753".
the Y of the a73r2 is "1361013.970240167".

[create a74r2]
a74r2 is a area. "An area (74) in r2".
the printed name of the a74r2 is "Room 2".
Understand "Area 74 in Room 2" as a74r2.
the parent of the a74r2 is r2.
the X of the a74r2 is "547088.7375940409".
the Y of the a74r2 is "1361006.3706751778".

[create a96r2]
a96r2 is a area. "An area (96) in r2".
the printed name of the a96r2 is "Room 2".
Understand "Area 96 in Room 2" as a96r2.
the parent of the a96r2 is r2.
the X of the a96r2 is "547077.8852593497".
the Y of the a96r2 is "1361038.3171190189".

[create a2r3]
a2r3 is a area. "An area (2) in r3".
the printed name of the a2r3 is "Room 3".
Understand "Area 2 in Room 3" as a2r3.
the parent of the a2r3 is r3.
the X of the a2r3 is "548485.648665368".
the Y of the a2r3 is "1362744.0114392212".

[create a3r3]
a3r3 is a area. "An area (3) in r3".
the printed name of the a3r3 is "Room 3".
Understand "Area 3 in Room 3" as a3r3.
the parent of the a3r3 is r3.
the X of the a3r3 is "548587.0980809142".
the Y of the a3r3 is "1362853.6222214063".

[create a4r3]
a4r3 is a area. "An area (4) in r3".
the printed name of the a4r3 is "Room 3".
Understand "Area 4 in Room 3" as a4r3.
the parent of the a4r3 is r3.
the X of the a4r3 is "548380.3482028818".
the Y of the a4r3 is "1362903.788113695".

[create a5r3]
a5r3 is a area. "An area (5) in r3".
the printed name of the a5r3 is "Room 3".
Understand "Area 5 in Room 3" as a5r3.
the parent of the a5r3 is r3.
the X of the a5r3 is "549844.4973304855".
the Y of the a5r3 is "1363011.8107591672".

[create a6r3]
a6r3 is a area. "An area (6) in r3".
the printed name of the a6r3 is "Room 3".
Understand "Area 6 in Room 3" as a6r3.
the parent of the a6r3 is r3.
the X of the a6r3 is "548361.4182296208".
the Y of the a6r3 is "1362996.4421486254".

[create a7r3]
a7r3 is a area. "An area (7) in r3".
the printed name of the a7r3 is "Room 3".
Understand "Area 7 in Room 3" as a7r3.
the parent of the a7r3 is r3.
the X of the a7r3 is "548416.7727272727".
the Y of the a7r3 is "1363081.0".

[create a17r3]
a17r3 is a area. "An area (17) in r3".
the printed name of the a17r3 is "Room 3".
Understand "Area 17 in Room 3" as a17r3.
the parent of the a17r3 is r3.
the X of the a17r3 is "548291.9864724483".
the Y of the a17r3 is "1362379.5489876661".

[create a20r3]
a20r3 is a area. "An area (20) in r3".
the printed name of the a20r3 is "Room 3".
Understand "Area 20 in Room 3" as a20r3.
the parent of the a20r3 is r3.
the X of the a20r3 is "546752.9032078378".
the Y of the a20r3 is "1362286.8473628089".

[create a41r3]
a41r3 is a area. "An area (41) in r3".
the printed name of the a41r3 is "Room 3".
Understand "Area 41 in Room 3" as a41r3.
the parent of the a41r3 is r3.
the X of the a41r3 is "546593.0".
the Y of the a41r3 is "1362327.0".

[create a42r3]
a42r3 is a area. "An area (42) in r3".
the printed name of the a42r3 is "Room 3".
Understand "Area 42 in Room 3" as a42r3.
the parent of the a42r3 is r3.
the X of the a42r3 is "546742.0294117648".
the Y of the a42r3 is "1362289.375".

[create a43r3]
a43r3 is a area. "An area (43) in r3".
the printed name of the a43r3 is "Room 3".
Understand "Area 43 in Room 3" as a43r3.
the parent of the a43r3 is r3.
the X of the a43r3 is "548120.0".
the Y of the a43r3 is "1362438.0".

[create a44r3]
a44r3 is a area. "An area (44) in r3".
the printed name of the a44r3 is "Room 3".
Understand "Area 44 in Room 3" as a44r3.
the parent of the a44r3 is r3.
the X of the a44r3 is "548234.7671002569".
the Y of the a44r3 is "1362398.5833333333".

[create a47r3]
a47r3 is a area. "An area (47) in r3".
the printed name of the a47r3 is "Room 3".
Understand "Area 47 in Room 3" as a47r3.
the parent of the a47r3 is r3.
the X of the a47r3 is "548507.0".
the Y of the a47r3 is "1362945.0".

[create a48r3]
a48r3 is a area. "An area (48) in r3".
the printed name of the a48r3 is "Room 3".
Understand "Area 48 in Room 3" as a48r3.
the parent of the a48r3 is r3.
the X of the a48r3 is "548584.4094621082".
the Y of the a48r3 is "1362857.4031760492".

[create a49r3]
a49r3 is a area. "An area (49) in r3".
the printed name of the a49r3 is "Room 3".
Understand "Area 49 in Room 3" as a49r3.
the parent of the a49r3 is r3.
the X of the a49r3 is "548452.0".
the Y of the a49r3 is "1362872.945736434".

[create a50r3]
a50r3 is a area. "An area (50) in r3".
the printed name of the a50r3 is "Room 3".
Understand "Area 50 in Room 3" as a50r3.
the parent of the a50r3 is r3.
the X of the a50r3 is "548448.9432626246".
the Y of the a50r3 is "1362903.7881136953".

[create a92r3]
a92r3 is a area. "An area (92) in r3".
the printed name of the a92r3 is "Room 3".
Understand "Area 92 in Room 3" as a92r3.
the parent of the a92r3 is r3.
the X of the a92r3 is "549877.8444444444".
the Y of the a92r3 is "1362997.0".

[create a93r3]
a93r3 is a area. "An area (93) in r3".
the printed name of the a93r3 is "Room 3".
Understand "Area 93 in Room 3" as a93r3.
the parent of the a93r3 is r3.
the X of the a93r3 is "548375.0".
the Y of the a93r3 is "1363086.0".

[create a0r4]
a0r4 is a area. "An area (0) in r4".
the printed name of the a0r4 is "Room 4".
Understand "Area 0 in Room 4" as a0r4.
the parent of the a0r4 is r4.
the X of the a0r4 is "549924.3288515309".
the Y of the a0r4 is "1362925.6828146877".

[create a1r4]
a1r4 is a area. "An area (1) in r4".
the printed name of the a1r4 is "Room 4".
Understand "Area 1 in Room 4" as a1r4.
the parent of the a1r4 is r4.
the X of the a1r4 is "550070.5359105705".
the Y of the a1r4 is "1362540.643192971".

[create a45r4]
a45r4 is a area. "An area (45) in r4".
the printed name of the a45r4 is "Room 4".
Understand "Area 45 in Room 4" as a45r4.
the parent of the a45r4 is r4.
the X of the a45r4 is "550233.0".
the Y of the a45r4 is "1362604.0".

[create a46r4]
a46r4 is a area. "An area (46) in r4".
the printed name of the a46r4 is "Room 4".
Understand "Area 46 in Room 4" as a46r4.
the parent of the a46r4 is r4.
the X of the a46r4 is "550138.497043806".
the Y of the a46r4 is "1362564.8333333335".

[create a91r4]
a91r4 is a area. "An area (91) in r4".
the printed name of the a91r4 is "Room 4".
Understand "Area 91 in Room 4" as a91r4.
the parent of the a91r4 is r4.
the X of the a91r4 is "549911.0".
the Y of the a91r4 is "1362960.31147541".

[create a91r5]
a91r5 is a area. "An area (91) in r5".
the printed name of the a91r5 is "Room 5".
Understand "Area 91 in Room 5" as a91r5.
the parent of the a91r5 is r5.
the X of the a91r5 is "549911.0".
the Y of the a91r5 is "1362960.31147541".

[create a92r5]
a92r5 is a area. "An area (92) in r5".
the printed name of the a92r5 is "Room 5".
Understand "Area 92 in Room 5" as a92r5.
the parent of the a92r5 is r5.
the X of the a92r5 is "549877.8444444444".
the Y of the a92r5 is "1362997.0".

[create d0]
d0 is a door. "A door between a19r1 and a21r2".
d0 is southeast of a19r1 and northwest of a21r2.

[create d2]
d2 is a door. "A door between a15r2 and a17r3".
d2 is north of a15r2 and south of a17r3.

[create d3]
d3 is a door. "A door between a23r2 and a20r3".
d3 is north of a23r2 and south of a20r3.

[create d4]
d4 is a door. "A door between a26r2 and a1r4".
d4 is north of a26r2 and south of a1r4.

[create d8]
d8 is a door. "A door between a18r0 and a19r1".
d8 is northeast of a18r0 and southwest of a19r1.

[create d9]
d9 is a door. "A door between a18r0 and a9r1".
d9 is north of a18r0 and south of a9r1.

south of a0r4 is north of a1r4.

northeast of a2r3 is southwest of a3r3.

northwest of a2r3 is southeast of a4r3.

east of a3r3 is west of a5r3.

north of a4r3 is south of a6r3.

northeast of a6r3 is southwest of a7r3.

east of a8r1 is west of a9r1.

east of a9r1 is west of a10r1.

east of a11r1 is west of a12r1.

east of a13r2 is west of a14r2.

south of a15r2 is north of a16r2.

southwest of a11r1 is northeast of a19r1.

southwest of a6r3 is northeast of a20r3.

southwest of a2r3 is northeast of a17r3.

southeast of a21r2 is northwest of a22r2.

southeast of a23r2 is northwest of a24r2.

south of a16r2 is north of a25r2.

southwest of a13r2 is northeast of a16r2.

west of a15r2 is east of a27r2.

northeast of a28r2 is southwest of a29r2.

southeast of a30r2 is northwest of a31r2.

northwest of a29r2 is southeast of a32r2.

northwest of a32r2 is southeast of a33r2.

southeast of a22r2 is northwest of a34r2.

southeast of a24r2 is northwest of a33r2.

south of a18r0 is north of a35r0.

south of a25r2 is north of a30r2.

west of a11r1 is east of a36r1.

southwest of a37r2 is northeast of a38r2.

southeast of a39r2 is northwest of a40r2.

east of a41r3 is west of a42r3.

east of a43r3 is west of a44r3.

southwest of a45r4 is northeast of a46r4.

southeast of a47r3 is northwest of a48r3.

north of a49r3 is south of a50r3.

northeast of a51r1 is southwest of a52r1.

northeast of a53r1 is southwest of a54r1.

southwest of a55r0 is northeast of a56r0.

northeast of a57r2 is southwest of a58r2.

northeast of a59r2 is southwest of a60r2.

east of a61r2 is west of a62r2.

west of a63r2 is east of a64r2.

west of a65r2 is east of a66r2.

northwest of a67r2 is southeast of a68r2.

northeast of a69r2 is southwest of a70r2.

northeast of a71r0 is southwest of a72r0.

west of a73r2 is east of a74r2.

southeast of a75r0 is northwest of a76r0.

northwest of a77r0 is southeast of a78r0.

northwest of a79r1 is southeast of a80r1.

northwest of a81r1 is southeast of a82r1.

southeast of a83r1 is northwest of a84r1.

southeast of a85r1 is northwest of a86r1.

southwest of a87r1 is northeast of a88r1.

southwest of a89r1 is northeast of a90r1.

northwest of a28r2 is southeast of a73r2.

north of a0r4 is south of a91r4.

southeast of a5r3 is northwest of a92r3.

west of a7r3 is east of a93r3.

northeast of a10r1 is southwest of a88r1.

northeast of a12r1 is southwest of a90r1.

northwest of a3r3 is southeast of a48r3.

east of a4r3 is west of a50r3.

southwest of a36r1 is northeast of a52r1.

southwest of a8r1 is northeast of a54r1.

southeast of a10r1 is northwest of a80r1.

southeast of a12r1 is northwest of a82r1.

northwest of a8r1 is southeast of a84r1.

northwest of a36r1 is southeast of a86r1.

southeast of a13r2 is northwest of a66r2.

southeast of a14r2 is northwest of a68r2.

southwest of a27r2 is northeast of a70r2.

northeast of a14r2 is southwest of a38r2.

northwest of a27r2 is southeast of a40r2.

west of a20r3 is east of a42r3.

west of a17r3 is east of a44r3.

east of a1r4 is west of a46r4.

southeast of a76r0 is northwest of a94r0.

northwest of a78r0 is southeast of a95r0.

southwest of a56r0 is northeast of a95r0.

south of a31r2 is north of a58r2.

southwest of a32r2 is northeast of a96r2.

southwest of a34r2 is northeast of a60r2.

west of a25r2 is east of a62r2.

east of a26r2 is west of a64r2.

northeast of a72r0 is southwest of a94r0.

north of a74r2 is south of a96r2.

east of a34r2 is west of a96r2.

east of a35r0 is west of a95r0.

west of a35r0 is east of a94r0.

southwest of a26r2 is northeast of a31r2.

east of a29r2 is west of a30r2.

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
the player is in a59r2.

the orientation of the player is 6.
