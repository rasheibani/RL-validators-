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

[create a0r0]
a0r0 is a area. "An area (0) in r0".
the printed name of the a0r0 is "Room 0".
Understand "Area 0 in Room 0" as a0r0.
the parent of the a0r0 is r0.
the X of the a0r0 is "547635.8469551243".
the Y of the a0r0 is "1387793.6063720852".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "548034.7817693059".
the Y of the a1r0 is "1387644.9748289345".

[create a2r0]
a2r0 is a area. "An area (2) in r0".
the printed name of the a2r0 is "Room 0".
Understand "Area 2 in Room 0" as a2r0.
the parent of the a2r0 is r0.
the X of the a2r0 is "545140.389199255".
the Y of the a2r0 is "1387640.2597765361".

[create a3r0]
a3r0 is a area. "An area (3) in r0".
the printed name of the a3r0 is "Room 0".
Understand "Area 3 in Room 0" as a3r0.
the parent of the a3r0 is r0.
the X of the a3r0 is "545522.0743049921".
the Y of the a3r0 is "1387767.884114292".

[create a4r0]
a4r0 is a area. "An area (4) in r0".
the printed name of the a4r0 is "Room 0".
Understand "Area 4 in Room 0" as a4r0.
the parent of the a4r0 is r0.
the X of the a4r0 is "546587.3134133653".
the Y of the a4r0 is "1387770.1574814427".

[create a5r0]
a5r0 is a area. "An area (5) in r0".
the printed name of the a5r0 is "Room 0".
Understand "Area 5 in Room 0" as a5r0.
the parent of the a5r0 is r0.
the X of the a5r0 is "547277.8226856526".
the Y of the a5r0 is "1387651.039730594".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "546194.0371580592".
the Y of the a6r0 is "1387653.3837988623".

[create a19r0]
a19r0 is a area. "An area (19) in r0".
the printed name of the a19r0 is "Room 0".
Understand "Area 19 in Room 0" as a19r0.
the parent of the a19r0 is r0.
the X of the a19r0 is "547674.5774167669".
the Y of the a19r0 is "1388538.5".

[create a20r0]
a20r0 is a area. "An area (20) in r0".
the printed name of the a20r0 is "Room 0".
Understand "Area 20 in Room 0" as a20r0.
the parent of the a20r0 is r0.
the X of the a20r0 is "547680.3882789807".
the Y of the a20r0 is "1388362.3609725628".

[create a25r0]
a25r0 is a area. "An area (25) in r0".
the printed name of the a25r0 is "Room 0".
Understand "Area 25 in Room 0" as a25r0.
the parent of the a25r0 is r0.
the X of the a25r0 is "546586.2412496979".
the Y of the a25r0 is "1388053.8284380112".

[create a26r0]
a26r0 is a area. "An area (26) in r0".
the printed name of the a26r0 is "Room 0".
Understand "Area 26 in Room 0" as a26r0.
the parent of the a26r0 is r0.
the X of the a26r0 is "547682.396597819".
the Y of the a26r0 is "1388002.3282346653".

[create a49r0]
a49r0 is a area. "An area (49) in r0".
the printed name of the a49r0 is "Room 0".
Understand "Area 49 in Room 0" as a49r0.
the parent of the a49r0 is r0.
the X of the a49r0 is "548083.0".
the Y of the a49r0 is "1387591.0".

[create a50r0]
a50r0 is a area. "An area (50) in r0".
the printed name of the a50r0 is "Room 0".
Understand "Area 50 in Room 0" as a50r0.
the parent of the a50r0 is r0.
the X of the a50r0 is "548051.7862838916".
the Y of the a50r0 is "1387616.6339712918".

[create a51r0]
a51r0 is a area. "An area (51) in r0".
the printed name of the a51r0 is "Room 0".
Understand "Area 51 in Room 0" as a51r0.
the parent of the a51r0 is r0.
the X of the a51r0 is "548092.0".
the Y of the a51r0 is "1387680.0".

[create a52r0]
a52r0 is a area. "An area (52) in r0".
the printed name of the a52r0 is "Room 0".
Understand "Area 52 in Room 0" as a52r0.
the parent of the a52r0 is r0.
the X of the a52r0 is "548049.1485252036".
the Y of the a52r0 is "1387661.8557671143".

[create a53r0]
a53r0 is a area. "An area (53) in r0".
the printed name of the a53r0 is "Room 0".
Understand "Area 53 in Room 0" as a53r0.
the parent of the a53r0 is r0.
the X of the a53r0 is "547240.0898876404".
the Y of the a53r0 is "1387600.0".

[create a54r0]
a54r0 is a area. "An area (54) in r0".
the printed name of the a54r0 is "Room 0".
Understand "Area 54 in Room 0" as a54r0.
the parent of the a54r0 is r0.
the X of the a54r0 is "547273.0417059886".
the Y of the a54r0 is "1387598.786189433".

[create a55r0]
a55r0 is a area. "An area (55) in r0".
the printed name of the a55r0 is "Room 0".
Understand "Area 55 in Room 0" as a55r0.
the parent of the a55r0 is r0.
the X of the a55r0 is "546156.0898876404".
the Y of the a55r0 is "1387600.0".

[create a56r0]
a56r0 is a area. "An area (56) in r0".
the printed name of the a56r0 is "Room 0".
Understand "Area 56 in Room 0" as a56r0.
the parent of the a56r0 is r0.
the X of the a56r0 is "546189.0417059885".
the Y of the a56r0 is "1387598.7861894332".

[create a57r0]
a57r0 is a area. "An area (57) in r0".
the printed name of the a57r0 is "Room 0".
Understand "Area 57 in Room 0" as a57r0.
the parent of the a57r0 is r0.
the X of the a57r0 is "547826.0161442359".
the Y of the a57r0 is "1387950.5096581134".

[create a58r0]
a58r0 is a area. "An area (58) in r0".
the printed name of the a58r0 is "Room 0".
Understand "Area 58 in Room 0" as a58r0.
the parent of the a58r0 is r0.
the X of the a58r0 is "547791.241025239".
the Y of the a58r0 is "1387994.4166666667".

[create a59r0]
a59r0 is a area. "An area (59) in r0".
the printed name of the a59r0 is "Room 0".
Understand "Area 59 in Room 0" as a59r0.
the parent of the a59r0 is r0.
the X of the a59r0 is "546719.0".
the Y of the a59r0 is "1388099.5903614457".

[create a60r0]
a60r0 is a area. "An area (60) in r0".
the printed name of the a60r0 is "Room 0".
Understand "Area 60 in Room 0" as a60r0.
the parent of the a60r0 is r0.
the X of the a60r0 is "546715.3168561121".
the Y of the a60r0 is "1388056.3708309203".

[create a61r0]
a61r0 is a area. "An area (61) in r0".
the printed name of the a61r0 is "Room 0".
Understand "Area 61 in Room 0" as a61r0.
the parent of the a61r0 is r0.
the X of the a61r0 is "547808.1533321161".
the Y of the a61r0 is "1388446.9469782729".

[create a62r0]
a62r0 is a area. "An area (62) in r0".
the printed name of the a62r0 is "Room 0".
Understand "Area 62 in Room 0" as a62r0.
the parent of the a62r0 is r0.
the X of the a62r0 is "547753.0285634215".
the Y of the a62r0 is "1388378.5833333337".

[create a65r0]
a65r0 is a area. "An area (65) in r0".
the printed name of the a65r0 is "Room 0".
Understand "Area 65 in Room 0" as a65r0.
the parent of the a65r0 is r0.
the X of the a65r0 is "545095.0".
the Y of the a65r0 is "1387591.0".

[create a66r0]
a66r0 is a area. "An area (66) in r0".
the printed name of the a66r0 is "Room 0".
Understand "Area 66 in Room 0" as a66r0.
the parent of the a66r0 is r0.
the X of the a66r0 is "545126.2137161086".
the Y of the a66r0 is "1387616.633971292".

[create a67r0]
a67r0 is a area. "An area (67) in r0".
the printed name of the a67r0 is "Room 0".
Understand "Area 67 in Room 0" as a67r0.
the parent of the a67r0 is r0.
the X of the a67r0 is "545086.0".
the Y of the a67r0 is "1387680.0".

[create a68r0]
a68r0 is a area. "An area (68) in r0".
the printed name of the a68r0 is "Room 0".
Understand "Area 68 in Room 0" as a68r0.
the parent of the a68r0 is r0.
the X of the a68r0 is "545117.0059757691".
the Y of the a68r0 is "1387660.657907662".

[create a7r1]
a7r1 is a area. "An area (7) in r1".
the printed name of the a7r1 is "Room 1".
Understand "Area 7 in Room 1" as a7r1.
the parent of the a7r1 is r1.
the X of the a7r1 is "547301.3067828342".
the Y of the a7r1 is "1389642.2601948914".

[create a8r1]
a8r1 is a area. "An area (8) in r1".
the printed name of the a8r1 is "Room 1".
Understand "Area 8 in Room 1" as a8r1.
the parent of the a8r1 is r1.
the X of the a8r1 is "547622.7329571336".
the Y of the a8r1 is "1389489.9626142487".

[create a9r1]
a9r1 is a area. "An area (9) in r1".
the printed name of the a9r1 is "Room 1".
Understand "Area 9 in Room 1" as a9r1.
the parent of the a9r1 is r1.
the X of the a9r1 is "546585.3445626479".
the Y of the a9r1 is "1389534.3503250591".

[create a10r1]
a10r1 is a area. "An area (10) in r1".
the printed name of the a10r1 is "Room 1".
Understand "Area 10 in Room 1" as a10r1.
the parent of the a10r1 is r1.
the X of the a10r1 is "547466.4238983432".
the Y of the a10r1 is "1388868.519271102".

[create a11r1]
a11r1 is a area. "An area (11) in r1".
the printed name of the a11r1 is "Room 1".
Understand "Area 11 in Room 1" as a11r1.
the parent of the a11r1 is r1.
the X of the a11r1 is "547607.3484625585".
the Y of the a11r1 is "1389010.5507250868".

[create a12r1]
a12r1 is a area. "An area (12) in r1".
the printed name of the a12r1 is "Room 1".
Understand "Area 12 in Room 1" as a12r1.
the parent of the a12r1 is r1.
the X of the a12r1 is "548686.7894678493".
the Y of the a12r1 is "1389312.9604600752".

[create a13r1]
a13r1 is a area. "An area (13) in r1".
the printed name of the a13r1 is "Room 1".
Understand "Area 13 in Room 1" as a13r1.
the parent of the a13r1 is r1.
the X of the a13r1 is "548687.3229297196".
the Y of the a13r1 is "1388941.76926021".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "548435.5663719161".
the Y of the a14r1 is "1389591.038212639".

[create a18r1]
a18r1 is a area. "An area (18) in r1".
the printed name of the a18r1 is "Room 1".
Understand "Area 18 in Room 1" as a18r1.
the parent of the a18r1 is r1.
the X of the a18r1 is "546227.0926719012".
the Y of the a18r1 is "1389659.8065726866".

[create a21r1]
a21r1 is a area. "An area (21) in r1".
the printed name of the a21r1 is "Room 1".
Understand "Area 21 in Room 1" as a21r1.
the parent of the a21r1 is r1.
the X of the a21r1 is "547666.3672280161".
the Y of the a21r1 is "1388672.524691358".

[create a22r1]
a22r1 is a area. "An area (22) in r1".
the printed name of the a22r1 is "Room 1".
Understand "Area 22 in Room 1" as a22r1.
the parent of the a22r1 is r1.
the X of the a22r1 is "547284.0419668956".
the Y of the a22r1 is "1388728.02277555".

[create a23r1]
a23r1 is a area. "An area (23) in r1".
the printed name of the a23r1 is "Room 1".
Understand "Area 23 in Room 1" as a23r1.
the parent of the a23r1 is r1.
the X of the a23r1 is "546586.3784065933".
the Y of the a23r1 is "1388679.5".

[create a24r1]
a24r1 is a area. "An area (24) in r1".
the printed name of the a24r1 is "Room 1".
Understand "Area 24 in Room 1" as a24r1.
the parent of the a24r1 is r1.
the X of the a24r1 is "547667.3548115535".
the Y of the a24r1 is "1388796.1576945162".

[create a27r1]
a27r1 is a area. "An area (27) in r1".
the printed name of the a27r1 is "Room 1".
Understand "Area 27 in Room 1" as a27r1.
the parent of the a27r1 is r1.
the X of the a27r1 is "546179.0".
the Y of the a27r1 is "1389714.0".

[create a28r1]
a28r1 is a area. "An area (28) in r1".
the printed name of the a28r1 is "Room 1".
Understand "Area 28 in Room 1" as a28r1.
the parent of the a28r1 is r1.
the X of the a28r1 is "546209.9153780069".
the Y of the a28r1 is "1389688.1834621995".

[create a31r1]
a31r1 is a area. "An area (31) in r1".
the printed name of the a31r1 is "Room 1".
Understand "Area 31 in Room 1" as a31r1.
the parent of the a31r1 is r1.
the X of the a31r1 is "547263.0".
the Y of the a31r1 is "1389714.0".

[create a32r1]
a32r1 is a area. "An area (32) in r1".
the printed name of the a32r1 is "Room 1".
Understand "Area 32 in Room 1" as a32r1.
the parent of the a32r1 is r1.
the X of the a32r1 is "547294.8264604809".
the Y of the a32r1 is "1389705.767353952".

[create a33r1]
a33r1 is a area. "An area (33) in r1".
the printed name of the a33r1 is "Room 1".
Understand "Area 33 in Room 1" as a33r1.
the parent of the a33r1 is r1.
the X of the a33r1 is "547524.0".
the Y of the a33r1 is "1388817.0".

[create a34r1]
a34r1 is a area. "An area (34) in r1".
the printed name of the a34r1 is "Room 1".
Understand "Area 34 in Room 1" as a34r1.
the parent of the a34r1 is r1.
the X of the a34r1 is "547472.5349209957".
the Y of the a34r1 is "1388858.3120970805".

[create a35r1]
a35r1 is a area. "An area (35) in r1".
the printed name of the a35r1 is "Room 1".
Understand "Area 35 in Room 1" as a35r1.
the parent of the a35r1 is r1.
the X of the a35r1 is "548836.0".
the Y of the a35r1 is "1388878.0".

[create a36r1]
a36r1 is a area. "An area (36) in r1".
the printed name of the a36r1 is "Room 1".
Understand "Area 36 in Room 1" as a36r1.
the parent of the a36r1 is r1.
the X of the a36r1 is "548749.8956183491".
the Y of the a36r1 is "1388913.7857142857".

[create a37r1]
a37r1 is a area. "An area (37) in r1".
the printed name of the a37r1 is "Room 1".
Understand "Area 37 in Room 1" as a37r1.
the parent of the a37r1 is r1.
the X of the a37r1 is "548836.0".
the Y of the a37r1 is "1389379.0".

[create a38r1]
a38r1 is a area. "An area (38) in r1".
the printed name of the a38r1 is "Room 1".
Understand "Area 38 in Room 1" as a38r1.
the parent of the a38r1 is r1.
the X of the a38r1 is "548756.7885154061".
the Y of the a38r1 is "1389343.214285714".

[create a39r1]
a39r1 is a area. "An area (39) in r1".
the printed name of the a39r1 is "Room 1".
Understand "Area 39 in Room 1" as a39r1.
the parent of the a39r1 is r1.
the X of the a39r1 is "548479.0".
the Y of the a39r1 is "1389704.0".

[create a40r1]
a40r1 is a area. "An area (40) in r1".
the printed name of the a40r1 is "Room 1".
Understand "Area 40 in Room 1" as a40r1.
the parent of the a40r1 is r1.
the X of the a40r1 is "548445.2".
the Y of the a40r1 is "1389615.6469450549".

[create a43r1]
a43r1 is a area. "An area (43) in r1".
the printed name of the a43r1 is "Room 1".
Understand "Area 43 in Room 1" as a43r1.
the parent of the a43r1 is r1.
the X of the a43r1 is "546170.0".
the Y of the a43r1 is "1389624.0".

[create a44r1]
a44r1 is a area. "An area (44) in r1".
the printed name of the a44r1 is "Room 1".
Understand "Area 44 in Room 1" as a44r1.
the parent of the a44r1 is r1.
the X of the a44r1 is "546212.06".
the Y of the a44r1 is "1389642.519".

[create a63r1]
a63r1 is a area. "An area (63) in r1".
the printed name of the a63r1 is "Room 1".
Understand "Area 63 in Room 1" as a63r1.
the parent of the a63r1 is r1.
the X of the a63r1 is "547530.0".
the Y of the a63r1 is "1388825.8604651163".

[create a64r1]
a64r1 is a area. "An area (64) in r1".
the printed name of the a64r1 is "Room 1".
Understand "Area 64 in Room 1" as a64r1.
the parent of the a64r1 is r1.
the X of the a64r1 is "547564.3210736206".
the Y of the a64r1 is "1388796.1576945165".

[create a15r2]
a15r2 is a area. "An area (15) in r2".
the printed name of the a15r2 is "Room 2".
Understand "Area 15 in Room 2" as a15r2.
the parent of the a15r2 is r2.
the X of the a15r2 is "545168.0926719012".
the Y of the a15r2 is "1389659.8065726864".

[create a16r2]
a16r2 is a area. "An area (16) in r2".
the printed name of the a16r2 is "Room 2".
Understand "Area 16 in Room 2" as a16r2.
the parent of the a16r2 is r2.
the X of the a16r2 is "545525.0".
the Y of the a16r2 is "1389535.3372641508".

[create a17r2]
a17r2 is a area. "An area (17) in r2".
the printed name of the a17r2 is "Room 2".
Understand "Area 17 in Room 2" as a17r2.
the parent of the a17r2 is r2.
the X of the a17r2 is "545882.8606168448".
the Y of the a17r2 is "1389659.8602906289".

[create a29r2]
a29r2 is a area. "An area (29) in r2".
the printed name of the a29r2 is "Room 2".
Understand "Area 29 in Room 2" as a29r2.
the parent of the a29r2 is r2.
the X of the a29r2 is "545931.0".
the Y of the a29r2 is "1389714.0".

[create a30r2]
a30r2 is a area. "An area (30) in r2".
the printed name of the a30r2 is "Room 2".
Understand "Area 30 in Room 2" as a30r2.
the parent of the a30r2 is r2.
the X of the a30r2 is "545899.9123931625".
the Y of the a30r2 is "1389688.1662393163".

[create a41r2]
a41r2 is a area. "An area (41) in r2".
the printed name of the a41r2 is "Room 2".
Understand "Area 41 in Room 2" as a41r2.
the parent of the a41r2 is r2.
the X of the a41r2 is "545111.0".
the Y of the a41r2 is "1389624.0".

[create a42r2]
a42r2 is a area. "An area (42) in r2".
the printed name of the a42r2 is "Room 2".
Understand "Area 42 in Room 2" as a42r2.
the parent of the a42r2 is r2.
the X of the a42r2 is "545153.0599999999".
the Y of the a42r2 is "1389642.5189999999".

[create a45r2]
a45r2 is a area. "An area (45) in r2".
the printed name of the a45r2 is "Room 2".
Understand "Area 45 in Room 2" as a45r2.
the parent of the a45r2 is r2.
the X of the a45r2 is "545940.0".
the Y of the a45r2 is "1389624.0".

[create a46r2]
a46r2 is a area. "An area (46) in r2".
the printed name of the a46r2 is "Room 2".
Understand "Area 46 in Room 2" as a46r2.
the parent of the a46r2 is r2.
the X of the a46r2 is "545897.94".
the Y of the a46r2 is "1389642.519".

[create a47r2]
a47r2 is a area. "An area (47) in r2".
the printed name of the a47r2 is "Room 2".
Understand "Area 47 in Room 2" as a47r2.
the parent of the a47r2 is r2.
the X of the a47r2 is "545120.0".
the Y of the a47r2 is "1389714.0".

[create a48r2]
a48r2 is a area. "An area (48) in r2".
the printed name of the a48r2 is "Room 2".
Understand "Area 48 in Room 2" as a48r2.
the parent of the a48r2 is r2.
the X of the a48r2 is "545150.9153780071".
the Y of the a48r2 is "1389688.1834621993".

[create d0]
d0 is a door. "A door between a3r0 and a16r2".
d0 is north of a3r0 and south of a16r2.

[create d2]
d2 is a door. "A door between a19r0 and a21r1".
d2 is north of a19r0 and south of a21r1.

[create d3]
d3 is a door. "A door between a25r0 and a23r1".
d3 is north of a25r0 and south of a23r1.

[create d4]
d4 is a door. "A door between a19r0 and a13r1".
d4 is east of a19r0 and west of a13r1.

east of a0r0 is west of a1r0.

east of a2r0 is west of a3r0.

east of a4r0 is west of a5r0.

east of a3r0 is west of a6r0.

southeast of a7r1 is northwest of a8r1.

west of a7r1 is east of a9r1.

northeast of a10r1 is southwest of a11r1.

south of a12r1 is north of a13r1.

south of a8r1 is north of a11r1.

northwest of a12r1 is southeast of a14r1.

east of a15r2 is west of a16r2.

east of a16r2 is west of a17r2.

west of a9r1 is east of a18r1.

east of a8r1 is west of a14r1.

south of a19r0 is north of a20r0.

west of a21r1 is east of a22r1.

west of a22r1 is east of a23r1.

southwest of a10r1 is northeast of a22r1.

north of a21r1 is south of a24r1.

south of a9r1 is north of a23r1.

south of a11r1 is north of a24r1.

west of a4r0 is east of a6r0.

west of a0r0 is east of a5r0.

north of a4r0 is south of a25r0.

north of a0r0 is south of a26r0.

south of a20r0 is north of a26r0.

southeast of a27r1 is northwest of a28r1.

southwest of a29r2 is northeast of a30r2.

east of a31r1 is west of a32r1.

northwest of a33r1 is southeast of a34r1.

northwest of a35r1 is southeast of a36r1.

southwest of a37r1 is northeast of a38r1.

south of a39r1 is north of a40r1.

northeast of a41r2 is southwest of a42r2.

northeast of a43r1 is southwest of a44r1.

northwest of a45r2 is southeast of a46r2.

southeast of a47r2 is northwest of a48r2.

northwest of a49r0 is southeast of a50r0.

southwest of a51r0 is northeast of a52r0.

east of a53r0 is west of a54r0.

east of a55r0 is west of a56r0.

northwest of a57r0 is southeast of a58r0.

south of a59r0 is north of a60r0.

southwest of a61r0 is northeast of a62r0.

southeast of a63r1 is northwest of a64r1.

northeast of a65r0 is southwest of a66r0.

southeast of a67r0 is northwest of a68r0.

northeast of a17r2 is southwest of a30r2.

north of a7r1 is south of a32r1.

southeast of a13r1 is northwest of a36r1.

northeast of a12r1 is southwest of a38r1.

north of a14r1 is south of a40r1.

southwest of a15r2 is northeast of a42r2.

southwest of a18r1 is northeast of a44r1.

southeast of a17r2 is northwest of a46r2.

northwest of a15r2 is southeast of a48r2.

northwest of a18r1 is southeast of a28r1.

northeast of a1r0 is southwest of a52r0.

south of a5r0 is north of a54r0.

south of a6r0 is north of a56r0.

east of a26r0 is west of a58r0.

east of a25r0 is west of a60r0.

east of a20r0 is west of a62r0.

west of a24r1 is east of a64r1.

southeast of a10r1 is northwest of a34r1.

southwest of a2r0 is northeast of a66r0.

northwest of a2r0 is southeast of a68r0.

southeast of a1r0 is northwest of a50r0.

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
the player is in a51r0.

the orientation of the player is 6.
