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

[create a34r0]
a34r0 is a area. "An area (34) in r0".
the printed name of the a34r0 is "Room 0".
Understand "Area 34 in Room 0" as a34r0.
the parent of the a34r0 is r0.
the X of the a34r0 is "548121.0".
the Y of the a34r0 is "1201363.0".

[create a35r0]
a35r0 is a area. "An area (35) in r0".
the printed name of the a35r0 is "Room 0".
Understand "Area 35 in Room 0" as a35r0.
the parent of the a35r0 is r0.
the X of the a35r0 is "548157.0".
the Y of the a35r0 is "1201455.3867768594".

[create a36r0]
a36r0 is a area. "An area (36) in r0".
the printed name of the a36r0 is "Room 0".
Understand "Area 36 in Room 0" as a36r0.
the parent of the a36r0 is r0.
the X of the a36r0 is "549365.0".
the Y of the a36r0 is "1201504.0".

[create a37r0]
a37r0 is a area. "An area (37) in r0".
the printed name of the a37r0 is "Room 0".
Understand "Area 37 in Room 0" as a37r0.
the parent of the a37r0 is r0.
the X of the a37r0 is "549312.4502168536".
the Y of the a37r0 is "1201548.7548575378".

[create a42r0]
a42r0 is a area. "An area (42) in r0".
the printed name of the a42r0 is "Room 0".
Understand "Area 42 in Room 0" as a42r0.
the parent of the a42r0 is r0.
the X of the a42r0 is "549436.0".
the Y of the a42r0 is "1201916.0".

[create a43r0]
a43r0 is a area. "An area (43) in r0".
the printed name of the a43r0 is "Room 0".
Understand "Area 43 in Room 0" as a43r0.
the parent of the a43r0 is r0.
the X of the a43r0 is "549402.3349056606".
the Y of the a43r0 is "1201896.278301887".

[create a44r0]
a44r0 is a area. "An area (44) in r0".
the printed name of the a44r0 is "Room 0".
Understand "Area 44 in Room 0" as a44r0.
the parent of the a44r0 is r0.
the X of the a44r0 is "549353.0".
the Y of the a44r0 is "1201953.0".

[create a45r0]
a45r0 is a area. "An area (45) in r0".
the printed name of the a45r0 is "Room 0".
Understand "Area 45 in Room 0" as a45r0.
the parent of the a45r0 is r0.
the X of the a45r0 is "549360.4335374578".
the Y of the a45r0 is "1201913.8779353779".

[create a46r0]
a46r0 is a area. "An area (46) in r0".
the printed name of the a46r0 is "Room 0".
Understand "Area 46 in Room 0" as a46r0.
the parent of the a46r0 is r0.
the X of the a46r0 is "547531.0".
the Y of the a46r0 is "1201913.0".

[create a47r0]
a47r0 is a area. "An area (47) in r0".
the printed name of the a47r0 is "Room 0".
Understand "Area 47 in Room 0" as a47r0.
the parent of the a47r0 is r0.
the X of the a47r0 is "547616.1526016354".
the Y of the a47r0 is "1201950.5769230768".

[create a72r0]
a72r0 is a area. "An area (72) in r0".
the printed name of the a72r0 is "Room 0".
Understand "Area 72 in Room 0" as a72r0.
the parent of the a72r0 is r0.
the X of the a72r0 is "547728.3872800174".
the Y of the a72r0 is "1201998.6766036483".

[create a78r0]
a78r0 is a area. "An area (78) in r0".
the printed name of the a78r0 is "Room 0".
Understand "Area 78 in Room 0" as a78r0.
the parent of the a78r0 is r0.
the X of the a78r0 is "548174.3005528573".
the Y of the a78r0 is "1201498.8583478576".

[create a79r0]
a79r0 is a area. "An area (79) in r0".
the printed name of the a79r0 is "Room 0".
Understand "Area 79 in Room 0" as a79r0.
the parent of the a79r0 is r0.
the X of the a79r0 is "549255.7849280777".
the Y of the a79r0 is "1201590.0136747023".

[create a81r0]
a81r0 is a area. "An area (81) in r0".
the printed name of the a81r0 is "Room 0".
Understand "Area 81 in Room 0" as a81r0.
the parent of the a81r0 is r0.
the X of the a81r0 is "549380.2748579546".
the Y of the a81r0 is "1201888.9249526516".

[create a0r1]
a0r1 is a area. "An area (0) in r1".
the printed name of the a0r1 is "Room 1".
Understand "Area 0 in Room 1" as a0r1.
the parent of the a0r1 is r1.
the X of the a0r1 is "546668.0".
the Y of the a0r1 is "1203025.0".

[create a1r1]
a1r1 is a area. "An area (1) in r1".
the printed name of the a1r1 is "Room 1".
Understand "Area 1 in Room 1" as a1r1.
the parent of the a1r1 is r1.
the X of the a1r1 is "546630.1025414974".
the Y of the a1r1 is "1203054.215920425".

[create a2r1]
a2r1 is a area. "An area (2) in r1".
the printed name of the a2r1 is "Room 1".
Understand "Area 2 in Room 1" as a2r1.
the parent of the a2r1 is r1.
the X of the a2r1 is "549270.0".
the Y of the a2r1 is "1202988.0".

[create a3r1]
a3r1 is a area. "An area (3) in r1".
the printed name of the a3r1 is "Room 1".
Understand "Area 3 in Room 1" as a3r1.
the parent of the a3r1 is r1.
the X of the a3r1 is "549277.4890701753".
the Y of the a3r1 is "1203026.997491228".

[create a4r1]
a4r1 is a area. "An area (4) in r1".
the printed name of the a4r1 is "Room 1".
Understand "Area 4 in Room 1" as a4r1.
the parent of the a4r1 is r1.
the X of the a4r1 is "549353.0".
the Y of the a4r1 is "1203025.0".

[create a5r1]
a5r1 is a area. "An area (5) in r1".
the printed name of the a5r1 is "Room 1".
Understand "Area 5 in Room 1" as a5r1.
the parent of the a5r1 is r1.
the X of the a5r1 is "549315.1025414974".
the Y of the a5r1 is "1203054.2159204248".

[create a6r1]
a6r1 is a area. "An area (6) in r1".
the printed name of the a6r1 is "Room 1".
Understand "Area 6 in Room 1" as a6r1.
the parent of the a6r1 is r1.
the X of the a6r1 is "547956.0718476252".
the Y of the a6r1 is "1202997.3934742426".

[create a7r1]
a7r1 is a area. "An area (7) in r1".
the printed name of the a7r1 is "Room 1".
Understand "Area 7 in Room 1" as a7r1.
the parent of the a7r1 is r1.
the X of the a7r1 is "547969.4784457928".
the Y of the a7r1 is "1203050.5824310577".

[create a8r1]
a8r1 is a area. "An area (8) in r1".
the printed name of the a8r1 is "Room 1".
Understand "Area 8 in Room 1" as a8r1.
the parent of the a8r1 is r1.
the X of the a8r1 is "547982.0".
the Y of the a8r1 is "1203385.7910958903".

[create a9r1]
a9r1 is a area. "An area (9) in r1".
the printed name of the a9r1 is "Room 1".
Understand "Area 9 in Room 1" as a9r1.
the parent of the a9r1 is r1.
the X of the a9r1 is "548005.8881112393".
the Y of the a9r1 is "1203408.5933219176".

[create a10r1]
a10r1 is a area. "An area (10) in r1".
the printed name of the a10r1 is "Room 1".
Understand "Area 10 in Room 1" as a10r1.
the parent of the a10r1 is r1.
the X of the a10r1 is "545108.0".
the Y of the a10r1 is "1203424.0".

[create a11r1]
a11r1 is a area. "An area (11) in r1".
the printed name of the a11r1 is "Room 1".
Understand "Area 11 in Room 1" as a11r1.
the parent of the a11r1 is r1.
the X of the a11r1 is "545150.06".
the Y of the a11r1 is "1203442.5189999999".

[create a12r1]
a12r1 is a area. "An area (12) in r1".
the printed name of the a12r1 is "Room 1".
Understand "Area 12 in Room 1" as a12r1.
the parent of the a12r1 is r1.
the X of the a12r1 is "545117.0".
the Y of the a12r1 is "1203514.0".

[create a13r1]
a13r1 is a area. "An area (13) in r1".
the printed name of the a13r1 is "Room 1".
Understand "Area 13 in Room 1" as a13r1.
the parent of the a13r1 is r1.
the X of the a13r1 is "545147.9153780066".
the Y of the a13r1 is "1203488.1834621995".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "548115.0".
the Y of the a14r1 is "1203532.0".

[create a15r1]
a15r1 is a area. "An area (15) in r1".
the printed name of the a15r1 is "Room 1".
Understand "Area 15 in Room 1" as a15r1.
the parent of the a15r1 is r1.
the X of the a15r1 is "548151.2142857143".
the Y of the a15r1 is "1203454.5251885823".

[create a20r1]
a20r1 is a area. "An area (20) in r1".
the printed name of the a20r1 is "Room 1".
Understand "Area 20 in Room 1" as a20r1.
the parent of the a20r1 is r1.
the X of the a20r1 is "546392.0".
the Y of the a20r1 is "1202810.0".

[create a21r1]
a21r1 is a area. "An area (21) in r1".
the printed name of the a21r1 is "Room 1".
Understand "Area 21 in Room 1" as a21r1.
the parent of the a21r1 is r1.
the X of the a21r1 is "546361.7769924194".
the Y of the a21r1 is "1202783.9929317762".

[create a22r1]
a22r1 is a area. "An area (22) in r1".
the printed name of the a22r1 is "Room 1".
Understand "Area 22 in Room 1" as a22r1.
the parent of the a22r1 is r1.
the X of the a22r1 is "546284.0".
the Y of the a22r1 is "1202822.0".

[create a23r1]
a23r1 is a area. "An area (23) in r1".
the printed name of the a23r1 is "Room 1".
Understand "Area 23 in Room 1" as a23r1.
the parent of the a23r1 is r1.
the X of the a23r1 is "546307.7621979022".
the Y of the a23r1 is "1202789.859781122".

[create a24r1]
a24r1 is a area. "An area (24) in r1".
the printed name of the a24r1 is "Room 1".
Understand "Area 24 in Room 1" as a24r1.
the parent of the a24r1 is r1.
the X of the a24r1 is "547531.0".
the Y of the a24r1 is "1202890.0".

[create a25r1]
a25r1 is a area. "An area (25) in r1".
the printed name of the a25r1 is "Room 1".
Understand "Area 25 in Room 1" as a25r1.
the parent of the a25r1 is r1.
the X of the a25r1 is "547622.9051178661".
the Y of the a25r1 is "1202852.4230769232".

[create a26r1]
a26r1 is a area. "An area (26) in r1".
the printed name of the a26r1 is "Room 1".
Understand "Area 26 in Room 1" as a26r1.
the parent of the a26r1 is r1.
the X of the a26r1 is "546114.0".
the Y of the a26r1 is "1203025.0".

[create a27r1]
a27r1 is a area. "An area (27) in r1".
the printed name of the a27r1 is "Room 1".
Understand "Area 27 in Room 1" as a27r1.
the parent of the a27r1 is r1.
the X of the a27r1 is "546151.8974585026".
the Y of the a27r1 is "1203054.2159204246".

[create a28r1]
a28r1 is a area. "An area (28) in r1".
the printed name of the a28r1 is "Room 1".
Understand "Area 28 in Room 1" as a28r1.
the parent of the a28r1 is r1.
the X of the a28r1 is "546197.0".
the Y of the a28r1 is "1202988.0".

[create a29r1]
a29r1 is a area. "An area (29) in r1".
the printed name of the a29r1 is "Room 1".
Understand "Area 29 in Room 1" as a29r1.
the parent of the a29r1 is r1.
the X of the a29r1 is "546188.7512146088".
the Y of the a29r1 is "1203025.2932652035".

[create a30r1]
a30r1 is a area. "An area (30) in r1".
the printed name of the a30r1 is "Room 1".
Understand "Area 30 in Room 1" as a30r1.
the parent of the a30r1 is r1.
the X of the a30r1 is "546585.0".
the Y of the a30r1 is "1202988.0".

[create a31r1]
a31r1 is a area. "An area (31) in r1".
the printed name of the a31r1 is "Room 1".
Understand "Area 31 in Room 1" as a31r1.
the parent of the a31r1 is r1.
the X of the a31r1 is "546593.2487853912".
the Y of the a31r1 is "1203025.2932652037".

[create a56r1]
a56r1 is a area. "An area (56) in r1".
the printed name of the a56r1 is "Room 1".
Understand "Area 56 in Room 1" as a56r1.
the parent of the a56r1 is r1.
the X of the a56r1 is "546151.0".
the Y of the a56r1 is "1203504.0".

[create a57r1]
a57r1 is a area. "An area (57) in r1".
the printed name of the a57r1 is "Room 1".
Understand "Area 57 in Room 1" as a57r1.
the parent of the a57r1 is r1.
the X of the a57r1 is "546180.4542705971".
the Y of the a57r1 is "1203513.5".

[create a58r1]
a58r1 is a area. "An area (58) in r1".
the printed name of the a58r1 is "Room 1".
Understand "Area 58 in Room 1" as a58r1.
the parent of the a58r1 is r1.
the X of the a58r1 is "546631.0".
the Y of the a58r1 is "1203523.0".

[create a59r1]
a59r1 is a area. "An area (59) in r1".
the printed name of the a59r1 is "Room 1".
Understand "Area 59 in Room 1" as a59r1.
the parent of the a59r1 is r1.
the X of the a59r1 is "546601.5457294029".
the Y of the a59r1 is "1203513.5".

[create a60r1]
a60r1 is a area. "An area (60) in r1".
the printed name of the a60r1 is "Room 1".
Understand "Area 60 in Room 1" as a60r1.
the parent of the a60r1 is r1.
the X of the a60r1 is "547982.0".
the Y of the a60r1 is "1203523.0".

[create a61r1]
a61r1 is a area. "An area (61) in r1".
the printed name of the a61r1 is "Room 1".
Understand "Area 61 in Room 1" as a61r1.
the parent of the a61r1 is r1.
the X of the a61r1 is "547952.0550595238".
the Y of the a61r1 is "1203500.0".

[create a62r1]
a62r1 is a area. "An area (62) in r1".
the printed name of the a62r1 is "Room 1".
Understand "Area 62 in Room 1" as a62r1.
the parent of the a62r1 is r1.
the X of the a62r1 is "549316.0".
the Y of the a62r1 is "1203504.0".

[create a63r1]
a63r1 is a area. "An area (63) in r1".
the printed name of the a63r1 is "Room 1".
Understand "Area 63 in Room 1" as a63r1.
the parent of the a63r1 is r1.
the X of the a63r1 is "549282.9520547945".
the Y of the a63r1 is "1203481.0".

[create a64r1]
a64r1 is a area. "An area (64) in r1".
the printed name of the a64r1 is "Room 1".
Understand "Area 64 in Room 1" as a64r1.
the parent of the a64r1 is r1.
the X of the a64r1 is "548167.7234637535".
the Y of the a64r1 is "1203414.972678037".

[create a65r1]
a65r1 is a area. "An area (65) in r1".
the printed name of the a65r1 is "Room 1".
Understand "Area 65 in Room 1" as a65r1.
the parent of the a65r1 is r1.
the X of the a65r1 is "546198.3082388516".
the Y of the a65r1 is "1203439.5".

[create a66r1]
a66r1 is a area. "An area (66) in r1".
the printed name of the a66r1 is "Room 1".
Understand "Area 66 in Room 1" as a66r1.
the parent of the a66r1 is r1.
the X of the a66r1 is "546583.6917611493".
the Y of the a66r1 is "1203439.5".

[create a67r1]
a67r1 is a area. "An area (67) in r1".
the printed name of the a67r1 is "Room 1".
Understand "Area 67 in Room 1" as a67r1.
the parent of the a67r1 is r1.
the X of the a67r1 is "546617.2524656302".
the Y of the a67r1 is "1203058.4668907023".

[create a68r1]
a68r1 is a area. "An area (68) in r1".
the printed name of the a68r1 is "Room 1".
Understand "Area 68 in Room 1" as a68r1.
the parent of the a68r1 is r1.
the X of the a68r1 is "549302.2035569828".
the Y of the a68r1 is "1203058.4830703114".

[create a69r1]
a69r1 is a area. "An area (69) in r1".
the printed name of the a69r1 is "Room 1".
Understand "Area 69 in Room 1" as a69r1.
the parent of the a69r1 is r1.
the X of the a69r1 is "547883.7541594934".
the Y of the a69r1 is "1203096.9340430032".

[create a70r1]
a70r1 is a area. "An area (70) in r1".
the printed name of the a70r1 is "Room 1".
Understand "Area 70 in Room 1" as a70r1.
the parent of the a70r1 is r1.
the X of the a70r1 is "548089.2737175634".
the Y of the a70r1 is "1203332.7409946697".

[create a71r1]
a71r1 is a area. "An area (71) in r1".
the printed name of the a71r1 is "Room 1".
Understand "Area 71 in Room 1" as a71r1.
the parent of the a71r1 is r1.
the X of the a71r1 is "545165.092671901".
the Y of the a71r1 is "1203459.8065726862".

[create a74r1]
a74r1 is a area. "An area (74) in r1".
the printed name of the a74r1 is "Room 1".
Understand "Area 74 in Room 1" as a74r1.
the parent of the a74r1 is r1.
the X of the a74r1 is "546328.858357395".
the Y of the a74r1 is "1202760.983477703".

[create a75r1]
a75r1 is a area. "An area (75) in r1".
the printed name of the a75r1 is "Room 1".
Understand "Area 75 in Room 1" as a75r1.
the parent of the a75r1 is r1.
the X of the a75r1 is "547707.5400383363".
the Y of the a75r1 is "1202821.5290505874".

[create a76r1]
a76r1 is a area. "An area (76) in r1".
the printed name of the a76r1 is "Room 1".
Understand "Area 76 in Room 1" as a76r1.
the parent of the a76r1 is r1.
the X of the a76r1 is "546164.74753437".
the Y of the a76r1 is "1203058.466890702".

[create a87r1]
a87r1 is a area. "An area (87) in r1".
the printed name of the a87r1 is "Room 1".
Understand "Area 87 in Room 1" as a87r1.
the parent of the a87r1 is r1.
the X of the a87r1 is "547855.8901191556".
the Y of the a87r1 is "1203387.380155866".

[create a88r1]
a88r1 is a area. "An area (88) in r1".
the printed name of the a88r1 is "Room 1".
Understand "Area 88 in Room 1" as a88r1.
the parent of the a88r1 is r1.
the X of the a88r1 is "546276.8319181078".
the Y of the a88r1 is "1203387.623403077".

[create a89r1]
a89r1 is a area. "An area (89) in r1".
the printed name of the a89r1 is "Room 1".
Understand "Area 89 in Room 1" as a89r1.
the parent of the a89r1 is r1.
the X of the a89r1 is "546505.1680818922".
the Y of the a89r1 is "1203387.6234030768".

[create a90r1]
a90r1 is a area. "An area (90) in r1".
the printed name of the a90r1 is "Room 1".
Understand "Area 90 in Room 1" as a90r1.
the parent of the a90r1 is r1.
the X of the a90r1 is "547984.437594088".
the Y of the a90r1 is "1203252.7271282887".

[create a91r1]
a91r1 is a area. "An area (91) in r1".
the printed name of the a91r1 is "Room 1".
Understand "Area 91 in Room 1" as a91r1.
the parent of the a91r1 is r1.
the X of the a91r1 is "549232.1267511036".
the Y of the a91r1 is "1203400.9824409904".

[create a92r1]
a92r1 is a area. "An area (92) in r1".
the printed name of the a92r1 is "Room 1".
Understand "Area 92 in Room 1" as a92r1.
the parent of the a92r1 is r1.
the X of the a92r1 is "545524.8959624639".
the Y of the a92r1 is "1203333.584535911".

[create a93r1]
a93r1 is a area. "An area (93) in r1".
the printed name of the a93r1 is "Room 1".
Understand "Area 93 in Room 1" as a93r1.
the parent of the a93r1 is r1.
the X of the a93r1 is "547063.0".
the Y of the a93r1 is "1203340.9340175954".

[create a94r1]
a94r1 is a area. "An area (94) in r1".
the printed name of the a94r1 is "Room 1".
Understand "Area 94 in Room 1" as a94r1.
the parent of the a94r1 is r1.
the X of the a94r1 is "545526.8626086956".
the Y of the a94r1 is "1202466.0".

[create a95r1]
a95r1 is a area. "An area (95) in r1".
the printed name of the a95r1 is "Room 1".
Understand "Area 95 in Room 1" as a95r1.
the parent of the a95r1 is r1.
the X of the a95r1 is "546259.7255000002".
the Y of the a95r1 is "1202466.0".

[create a16r2]
a16r2 is a area. "An area (16) in r2".
the printed name of the a16r2 is "Room 2".
Understand "Area 16 in Room 2" as a16r2.
the parent of the a16r2 is r2.
the X of the a16r2 is "546284.0".
the Y of the a16r2 is "1202110.0".

[create a17r2]
a17r2 is a area. "An area (17) in r2".
the printed name of the a17r2 is "Room 2".
Understand "Area 17 in Room 2" as a17r2.
the parent of the a17r2 is r2.
the X of the a17r2 is "546307.0755088733".
the Y of the a17r2 is "1202142.0063361116".

[create a18r2]
a18r2 is a area. "An area (18) in r2".
the printed name of the a18r2 is "Room 2".
Understand "Area 18 in Room 2" as a18r2.
the parent of the a18r2 is r2.
the X of the a18r2 is "546392.0".
the Y of the a18r2 is "1202125.0".

[create a19r2]
a19r2 is a area. "An area (19) in r2".
the printed name of the a19r2 is "Room 2".
Understand "Area 19 in Room 2" as a19r2.
the parent of the a19r2 is r2.
the X of the a19r2 is "546360.8077975204".
the Y of the a19r2 is "1202151.4338578526".

[create a32r2]
a32r2 is a area. "An area (32) in r2".
the printed name of the a32r2 is "Room 2".
Understand "Area 32 in Room 2" as a32r2.
the parent of the a32r2 is r2.
the X of the a32r2 is "547487.0".
the Y of the a32r2 is "1201480.0".

[create a33r2]
a33r2 is a area. "An area (33) in r2".
the printed name of the a33r2 is "Room 2".
Understand "Area 33 in Room 2" as a33r2.
the parent of the a33r2 is r2.
the X of the a33r2 is "547443.9383292025".
the Y of the a33r2 is "1201461.8770228894".

[create a38r2]
a38r2 is a area. "An area (38) in r2".
the printed name of the a38r2 is "Room 2".
Understand "Area 38 in Room 2" as a38r2.
the parent of the a38r2 is r2.
the X of the a38r2 is "546763.0".
the Y of the a38r2 is "1201888.0".

[create a39r2]
a39r2 is a area. "An area (39) in r2".
the printed name of the a39r2 is "Room 2".
Understand "Area 39 in Room 2" as a39r2.
the parent of the a39r2 is r2.
the X of the a39r2 is "546729.1511123562".
the Y of the a39r2 is "1201867.8660087993".

[create a40r2]
a40r2 is a area. "An area (40) in r2".
the printed name of the a40r2 is "Room 2".
Understand "Area 40 in Room 2" as a40r2.
the parent of the a40r2 is r2.
the X of the a40r2 is "546680.0".
the Y of the a40r2 is "1201925.0".

[create a41r2]
a41r2 is a area. "An area (41) in r2".
the printed name of the a41r2 is "Room 2".
Understand "Area 41 in Room 2" as a41r2.
the parent of the a41r2 is r2.
the X of the a41r2 is "546686.5365348754".
the Y of the a41r2 is "1201883.8657403952".

[create a48r2]
a48r2 is a area. "An area (48) in r2".
the printed name of the a48r2 is "Room 2".
Understand "Area 48 in Room 2" as a48r2.
the parent of the a48r2 is r2.
the X of the a48r2 is "545092.0".
the Y of the a48r2 is "1201391.0".

[create a49r2]
a49r2 is a area. "An area (49) in r2".
the printed name of the a49r2 is "Room 2".
Understand "Area 49 in Room 2" as a49r2.
the parent of the a49r2 is r2.
the X of the a49r2 is "545123.2137161086".
the Y of the a49r2 is "1201416.6339712918".

[create a50r2]
a50r2 is a area. "An area (50) in r2".
the printed name of the a50r2 is "Room 2".
Understand "Area 50 in Room 2" as a50r2.
the parent of the a50r2 is r2.
the X of the a50r2 is "545083.0".
the Y of the a50r2 is "1201480.0".

[create a51r2]
a51r2 is a area. "An area (51) in r2".
the printed name of the a51r2 is "Room 2".
Understand "Area 51 in Room 2" as a51r2.
the parent of the a51r2 is r2.
the X of the a51r2 is "545114.0059757696".
the Y of the a51r2 is "1201460.657907662".

[create a52r2]
a52r2 is a area. "An area (52) in r2".
the printed name of the a52r2 is "Room 2".
Understand "Area 52 in Room 2" as a52r2.
the parent of the a52r2 is r2.
the X of the a52r2 is "546636.0".
the Y of the a52r2 is "1201391.0".

[create a53r2]
a53r2 is a area. "An area (53) in r2".
the printed name of the a53r2 is "Room 2".
Understand "Area 53 in Room 2" as a53r2.
the parent of the a53r2 is r2.
the X of the a53r2 is "546668.0417059885".
the Y of the a53r2 is "1201398.7861894327".

[create a54r2]
a54r2 is a area. "An area (54) in r2".
the printed name of the a54r2 is "Room 2".
Understand "Area 54 in Room 2" as a54r2.
the parent of the a54r2 is r2.
the X of the a54r2 is "547478.0".
the Y of the a54r2 is "1201391.0".

[create a55r2]
a55r2 is a area. "An area (55) in r2".
the printed name of the a55r2 is "Room 2".
Understand "Area 55 in Room 2" as a55r2.
the parent of the a55r2 is r2.
the X of the a55r2 is "547446.7862838916".
the Y of the a55r2 is "1201416.633971292".

[create a73r2]
a73r2 is a area. "An area (73) in r2".
the printed name of the a73r2 is "Room 2".
Understand "Area 73 in Room 2" as a73r2.
the parent of the a73r2 is r2.
the X of the a73r2 is "546328.9241359932".
the Y of the a73r2 is "1202172.8985341634".

[create a77r2]
a77r2 is a area. "An area (77) in r2".
the printed name of the a77r2 is "Room 2".
Understand "Area 77 in Room 2" as a77r2.
the parent of the a77r2 is r2.
the X of the a77r2 is "547429.7290448344".
the Y of the a77r2 is "1201445.0627030537".

[create a80r2]
a80r2 is a area. "An area (80) in r2".
the printed name of the a80r2 is "Room 2".
Understand "Area 80 in Room 2" as a80r2.
the parent of the a80r2 is r2.
the X of the a80r2 is "546705.3706921167".
the Y of the a80r2 is "1201860.504141509".

[create a82r2]
a82r2 is a area. "An area (82) in r2".
the printed name of the a82r2 is "Room 2".
Understand "Area 82 in Room 2" as a82r2.
the parent of the a82r2 is r2.
the X of the a82r2 is "545137.3891992552".
the Y of the a82r2 is "1201440.2597765364".

[create a83r2]
a83r2 is a area. "An area (83) in r2".
the printed name of the a83r2 is "Room 2".
Understand "Area 83 in Room 2" as a83r2.
the parent of the a83r2 is r2.
the X of the a83r2 is "546672.0304135778".
the Y of the a83r2 is "1201442.3806221308".

[create a84r2]
a84r2 is a area. "An area (84) in r2".
the printed name of the a84r2 is "Room 2".
Understand "Area 84 in Room 2" as a84r2.
the parent of the a84r2 is r2.
the X of the a84r2 is "547063.0".
the Y of the a84r2 is "1201568.5778301887".

[create a85r2]
a85r2 is a area. "An area (85) in r2".
the printed name of the a85r2 is "Room 2".
Understand "Area 85 in Room 2" as a85r2.
the parent of the a85r2 is r2.
the X of the a85r2 is "545524.6635164127".
the Y of the a85r2 is "1201572.5895158404".

[create a86r2]
a86r2 is a area. "An area (86) in r2".
the printed name of the a86r2 is "Room 2".
Understand "Area 86 in Room 2" as a86r2.
the parent of the a86r2 is r2.
the X of the a86r2 is "546580.2973178107".
the Y of the a86r2 is "1201514.4867492085".

[create d0]
d0 is a door. "A door between a94r1 and a85r2".
d0 is south of a94r1 and north of a85r2.

[create d1]
d1 is a door. "A door between a93r1 and a84r2".
d1 is south of a93r1 and north of a84r2.

[create d2]
d2 is a door. "A door between a95r1 and a73r2".
d2 is south of a95r1 and north of a73r2.

[create d4]
d4 is a door. "A door between a72r0 and a75r1".
d4 is north of a72r0 and south of a75r1.

northwest of a0r1 is southeast of a1r1.

north of a2r1 is south of a3r1.

northwest of a4r1 is southeast of a5r1.

north of a6r1 is south of a7r1.

northeast of a8r1 is southwest of a9r1.

northeast of a10r1 is southwest of a11r1.

southeast of a12r1 is northwest of a13r1.

southeast of a14r1 is northwest of a15r1.

northeast of a16r2 is southwest of a17r2.

northwest of a18r2 is southeast of a19r2.

southwest of a20r1 is northeast of a21r1.

southeast of a22r1 is northwest of a23r1.

east of a24r1 is west of a25r1.

northeast of a26r1 is southwest of a27r1.

north of a28r1 is south of a29r1.

north of a30r1 is south of a31r1.

southwest of a32r2 is northeast of a33r2.

north of a34r0 is south of a35r0.

northwest of a36r0 is southeast of a37r0.

southwest of a38r2 is northeast of a39r2.

south of a40r2 is north of a41r2.

southwest of a42r0 is northeast of a43r0.

south of a44r0 is north of a45r0.

northeast of a46r0 is southwest of a47r0.

northeast of a48r2 is southwest of a49r2.

southeast of a50r2 is northwest of a51r2.

east of a52r2 is west of a53r2.

northwest of a54r2 is southeast of a55r2.

east of a56r1 is west of a57r1.

west of a58r1 is east of a59r1.

southwest of a60r1 is northeast of a61r1.

southwest of a62r1 is northeast of a63r1.

southeast of a15r1 is northwest of a64r1.

south of a57r1 is north of a65r1.

south of a59r1 is north of a66r1.

northeast of a31r1 is southwest of a67r1.

west of a1r1 is east of a67r1.

northeast of a3r1 is southwest of a68r1.

west of a5r1 is east of a68r1.

northwest of a7r1 is southeast of a69r1.

southeast of a9r1 is northwest of a70r1.

northeast of a11r1 is southwest of a71r1.

southeast of a13r1 is northwest of a71r1.

northeast of a47r0 is southwest of a72r0.

northeast of a17r2 is southwest of a73r2.

northwest of a19r2 is southeast of a73r2.

southwest of a21r1 is northeast of a74r1.

southeast of a23r1 is northwest of a74r1.

east of a25r1 is west of a75r1.

east of a27r1 is west of a76r1.

northwest of a29r1 is southeast of a76r1.

northwest of a55r2 is southeast of a77r2.

southwest of a33r2 is northeast of a77r2.

north of a35r0 is south of a78r0.

northwest of a37r0 is southeast of a79r0.

west of a39r2 is east of a80r2.

southeast of a41r2 is northwest of a80r2.

west of a43r0 is east of a81r0.

southeast of a45r0 is northwest of a81r0.

northeast of a49r2 is southwest of a82r2.

southeast of a51r2 is northwest of a82r2.

north of a53r2 is south of a83r2.

west of a77r2 is east of a84r2.

east of a82r2 is west of a85r2.

east of a83r2 is west of a84r2.

east of a85r2 is west of a86r2.

east of a78r0 is west of a79r0.

northwest of a83r2 is southeast of a86r2.

south of a80r2 is north of a86r2.

southeast of a72r0 is northwest of a78r0.

southwest of a61r1 is northeast of a87r1.

southeast of a65r1 is northwest of a88r1.

southwest of a66r1 is northeast of a89r1.

southeast of a87r1 is northwest of a90r1.

southwest of a63r1 is northeast of a91r1.

east of a64r1 is west of a91r1.

west of a65r1 is east of a92r1.

east of a71r1 is west of a92r1.

west of a87r1 is east of a93r1.

east of a66r1 is west of a93r1.

east of a88r1 is west of a89r1.

south of a92r1 is north of a94r1.

southwest of a69r1 is northeast of a75r1.

north of a76r1 is south of a88r1.

north of a67r1 is south of a89r1.

north of a68r1 is south of a91r1.

northeast of a69r1 is southwest of a90r1.

southwest of a70r1 is northeast of a90r1.

southwest of a64r1 is northeast of a70r1.

northeast of a79r0 is southwest of a81r0.

east of a94r1 is west of a95r1.

south of a74r1 is north of a95r1.

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
the player is in a77r2.

the orientation of the player is 6.
