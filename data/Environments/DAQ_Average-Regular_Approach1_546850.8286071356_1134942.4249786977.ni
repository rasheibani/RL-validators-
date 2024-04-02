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

[create r7]
r7 is a indoor_room. "Room 7".

[create a20r0]
a20r0 is a area. "An area (20) in r0".
the printed name of the a20r0 is "Room 0".
Understand "Area 20 in Room 0" as a20r0.
the parent of the a20r0 is r0.
the X of the a20r0 is "548206.0".
the Y of the a20r0 is "1134438.0".

[create a21r0]
a21r0 is a area. "An area (21) in r0".
the printed name of the a21r0 is "Room 0".
Understand "Area 21 in Room 0" as a21r0.
the parent of the a21r0 is r0.
the X of the a21r0 is "548316.0394056847".
the Y of the a21r0 is "1134398.5833333335".

[create a55r0]
a55r0 is a area. "An area (55) in r0".
the printed name of the a55r0 is "Room 0".
Understand "Area 55 in Room 0" as a55r0.
the parent of the a55r0 is r0.
the X of the a55r0 is "549963.8444444444".
the Y of the a55r0 is "1134997.0".

[create a56r0]
a56r0 is a area. "An area (56) in r0".
the printed name of the a56r0 is "Room 0".
Understand "Area 56 in Room 0" as a56r0.
the parent of the a56r0 is r0.
the X of the a56r0 is "549930.5516899223".
the Y of the a56r0 is "1135011.7866160134".

[create a60r0]
a60r0 is a area. "An area (60) in r0".
the printed name of the a60r0 is "Room 0".
Understand "Area 60 in Room 0" as a60r0.
the parent of the a60r0 is r0.
the X of the a60r0 is "548377.8567402593".
the Y of the a60r0 is "1134378.9176461317".

[create a0r1]
a0r1 is a area. "An area (0) in r1".
the printed name of the a0r1 is "Room 1".
Understand "Area 0 in Room 1" as a0r1.
the parent of the a0r1 is r1.
the X of the a0r1 is "550258.0".
the Y of the a0r1 is "1132333.0".

[create a1r1]
a1r1 is a area. "An area (1) in r1".
the printed name of the a1r1 is "Room 1".
Understand "Area 1 in Room 1" as a1r1.
the parent of the a1r1 is r1.
the X of the a1r1 is "550262.2013157895".
the Y of the a1r1 is "1132406.8065789475".

[create a10r1]
a10r1 is a area. "An area (10) in r1".
the printed name of the a10r1 is "Room 1".
Understand "Area 10 in Room 1" as a10r1.
the parent of the a10r1 is r1.
the X of the a10r1 is "549938.0".
the Y of the a10r1 is "1133101.0".

[create a11r1]
a11r1 is a area. "An area (11) in r1".
the printed name of the a11r1 is "Room 1".
Understand "Area 11 in Room 1" as a11r1.
the parent of the a11r1 is r1.
the X of the a11r1 is "549882.9915114488".
the Y of the a11r1 is "1133177.2877526".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "548206.0".
the Y of the a14r1 is "1133716.1340782123".

[create a16r1]
a16r1 is a area. "An area (16) in r1".
the printed name of the a16r1 is "Room 1".
Understand "Area 16 in Room 1" as a16r1.
the parent of the a16r1 is r1.
the X of the a16r1 is "550319.0".
the Y of the a16r1 is "1133731.0".

[create a28r1]
a28r1 is a area. "An area (28) in r1".
the printed name of the a28r1 is "Room 1".
Understand "Area 28 in Room 1" as a28r1.
the parent of the a28r1 is r1.
the X of the a28r1 is "547804.0".
the Y of the a28r1 is "1132991.0".

[create a29r1]
a29r1 is a area. "An area (29) in r1".
the printed name of the a29r1 is "Room 1".
Understand "Area 29 in Room 1" as a29r1.
the parent of the a29r1 is r1.
the X of the a29r1 is "547835.2137161084".
the Y of the a29r1 is "1133016.633971292".

[create a34r1]
a34r1 is a area. "An area (34) in r1".
the printed name of the a34r1 is "Room 1".
Understand "Area 34 in Room 1" as a34r1.
the parent of the a34r1 is r1.
the X of the a34r1 is "547795.0".
the Y of the a34r1 is "1133080.0".

[create a35r1]
a35r1 is a area. "An area (35) in r1".
the printed name of the a35r1 is "Room 1".
Understand "Area 35 in Room 1" as a35r1.
the parent of the a35r1 is r1.
the X of the a35r1 is "547836.3556858187".
the Y of the a35r1 is "1133061.7045075544".

[create a36r1]
a36r1 is a area. "An area (36) in r1".
the printed name of the a36r1 is "Room 1".
Understand "Area 36 in Room 1" as a36r1.
the parent of the a36r1 is r1.
the X of the a36r1 is "548646.0".
the Y of the a36r1 is "1132991.0".

[create a37r1]
a37r1 is a area. "An area (37) in r1".
the printed name of the a37r1 is "Room 1".
Understand "Area 37 in Room 1" as a37r1.
the parent of the a37r1 is r1.
the X of the a37r1 is "548614.0315235816".
the Y of the a37r1 is "1133000.3646934978".

[create a40r1]
a40r1 is a area. "An area (40) in r1".
the printed name of the a40r1 is "Room 1".
Understand "Area 40 in Room 1" as a40r1.
the parent of the a40r1 is r1.
the X of the a40r1 is "548561.2081813444".
the Y of the a40r1 is "1133099.030224075".

[create a41r1]
a41r1 is a area. "An area (41) in r1".
the printed name of the a41r1 is "Room 1".
Understand "Area 41 in Room 1" as a41r1.
the parent of the a41r1 is r1.
the X of the a41r1 is "548543.7558590334".
the Y of the a41r1 is "1133126.4416846747".

[create a48r1]
a48r1 is a area. "An area (48) in r1".
the printed name of the a48r1 is "Room 1".
Understand "Area 48 in Room 1" as a48r1.
the parent of the a48r1 is r1.
the X of the a48r1 is "550513.0".
the Y of the a48r1 is "1132521.0".

[create a49r1]
a49r1 is a area. "An area (49) in r1".
the printed name of the a49r1 is "Room 1".
Understand "Area 49 in Room 1" as a49r1.
the parent of the a49r1 is r1.
the X of the a49r1 is "550458.0661764706".
the Y of the a49r1 is "1132487.0".

[create a50r1]
a50r1 is a area. "An area (50) in r1".
the printed name of the a50r1 is "Room 1".
Understand "Area 50 in Room 1" as a50r1.
the parent of the a50r1 is r1.
the X of the a50r1 is "548688.2188379406".
the Y of the a50r1 is "1132992.8626055287".

[create a65r1]
a65r1 is a area. "An area (65) in r1".
the printed name of the a65r1 is "Room 1".
Understand "Area 65 in Room 1" as a65r1.
the parent of the a65r1 is r1.
the X of the a65r1 is "548636.7708136316".
the Y of the a65r1 is "1133209.8815704172".

[create a66r1]
a66r1 is a area. "An area (66) in r1".
the printed name of the a66r1 is "Room 1".
Understand "Area 66 in Room 1" as a66r1.
the parent of the a66r1 is r1.
the X of the a66r1 is "548696.764465824".
the Y of the a66r1 is "1133134.1046847703".

[create a67r1]
a67r1 is a area. "An area (67) in r1".
the printed name of the a67r1 is "Room 1".
Understand "Area 67 in Room 1" as a67r1.
the parent of the a67r1 is r1.
the X of the a67r1 is "549851.0253397932".
the Y of the a67r1 is "1133217.9152977373".

[create a69r1]
a69r1 is a area. "An area (69) in r1".
the printed name of the a69r1 is "Room 1".
Understand "Area 69 in Room 1" as a69r1.
the parent of the a69r1 is r1.
the X of the a69r1 is "548200.2762648677".
the Y of the a69r1 is "1133498.4502509376".

[create a70r1]
a70r1 is a area. "An area (70) in r1".
the printed name of the a70r1 is "Room 1".
Understand "Area 70 in Room 1" as a70r1.
the parent of the a70r1 is r1.
the X of the a70r1 is "548367.9214640281".
the Y of the a70r1 is "1133560.967939791".

[create a71r1]
a71r1 is a area. "An area (71) in r1".
the printed name of the a71r1 is "Room 1".
Understand "Area 71 in Room 1" as a71r1.
the parent of the a71r1 is r1.
the X of the a71r1 is "548060.7223976618".
the Y of the a71r1 is "1133702.2767870517".

[create a75r1]
a75r1 is a area. "An area (75) in r1".
the printed name of the a75r1 is "Room 1".
Understand "Area 75 in Room 1" as a75r1.
the parent of the a75r1 is r1.
the X of the a75r1 is "547851.6344696969".
the Y of the a75r1 is "1133044.0018939395".

[create a77r1]
a77r1 is a area. "An area (77) in r1".
the printed name of the a77r1 is "Room 1".
Understand "Area 77 in Room 1" as a77r1.
the parent of the a77r1 is r1.
the X of the a77r1 is "548606.6890183382".
the Y of the a77r1 is "1133053.3017098198".

[create a79r1]
a79r1 is a area. "An area (79) in r1".
the printed name of the a79r1 is "Room 1".
Understand "Area 79 in Room 1" as a79r1.
the parent of the a79r1 is r1.
the X of the a79r1 is "550270.7952866211".
the Y of the a79r1 is "1132490.5977945551".

[create a82r1]
a82r1 is a area. "An area (82) in r1".
the printed name of the a82r1 is "Room 1".
Understand "Area 82 in Room 1" as a82r1.
the parent of the a82r1 is r1.
the X of the a82r1 is "548758.4679082602".
the Y of the a82r1 is "1133095.1916367898".

[create a84r1]
a84r1 is a area. "An area (84) in r1".
the printed name of the a84r1 is "Room 1".
Understand "Area 84 in Room 1" as a84r1.
the parent of the a84r1 is r1.
the X of the a84r1 is "548232.4310969368".
the Y of the a84r1 is "1133167.4370029136".

[create a87r1]
a87r1 is a area. "An area (87) in r1".
the printed name of the a87r1 is "Room 1".
Understand "Area 87 in Room 1" as a87r1.
the parent of the a87r1 is r1.
the X of the a87r1 is "549436.8480141503".
the Y of the a87r1 is "1132983.1248914616".

[create a17r2]
a17r2 is a area. "An area (17) in r2".
the printed name of the a17r2 is "Room 2".
Understand "Area 17 in Room 2" as a17r2.
the parent of the a17r2 is r2.
the X of the a17r2 is "550192.3318897632".
the Y of the a17r2 is "1133764.5833333335".

[create a22r2]
a22r2 is a area. "An area (22) in r2".
the printed name of the a22r2 is "Room 2".
Understand "Area 22 in Room 2" as a22r2.
the parent of the a22r2 is r2.
the X of the a22r2 is "550319.0".
the Y of the a22r2 is "1134604.0".

[create a23r2]
a23r2 is a area. "An area (23) in r2".
the printed name of the a23r2 is "Room 2".
Understand "Area 23 in Room 2" as a23r2.
the parent of the a23r2 is r2.
the X of the a23r2 is "550224.4970438055".
the Y of the a23r2 is "1134564.8333333333".

[create a57r2]
a57r2 is a area. "An area (57) in r2".
the printed name of the a57r2 is "Room 2".
Understand "Area 57 in Room 2" as a57r2.
the parent of the a57r2 is r2.
the X of the a57r2 is "549997.0".
the Y of the a57r2 is "1134960.31147541".

[create a58r2]
a58r2 is a area. "An area (58) in r2".
the printed name of the a58r2 is "Room 2".
Understand "Area 58 in Room 2" as a58r2.
the parent of the a58r2 is r2.
the X of the a58r2 is "550010.3079281528".
the Y of the a58r2 is "1134925.7371741314".

[create a61r2]
a61r2 is a area. "An area (61) in r2".
the printed name of the a61r2 is "Room 2".
Understand "Area 61 in Room 2" as a61r2.
the parent of the a61r2 is r2.
the X of the a61r2 is "550156.5359105705".
the Y of the a61r2 is "1134540.643192971".

[create a72r2]
a72r2 is a area. "An area (72) in r2".
the printed name of the a72r2 is "Room 2".
Understand "Area 72 in Room 2" as a72r2.
the parent of the a72r2 is r2.
the X of the a72r2 is "550154.3476716748".
the Y of the a72r2 is "1133775.701377194".

[create a6r3]
a6r3 is a area. "An area (6) in r3".
the printed name of the a6r3 is "Room 3".
Understand "Area 6 in Room 3" as a6r3.
the parent of the a6r3 is r3.
the X of the a6r3 is "547411.0".
the Y of the a6r3 is "1135101.0".

[create a7r3]
a7r3 is a area. "An area (7) in r3".
the printed name of the a7r3 is "Room 3".
Understand "Area 7 in Room 3" as a7r3.
the parent of the a7r3 is r3.
the X of the a7r3 is "547450.8234236795".
the Y of the a7r3 is "1135049.335350889".

[create a15r3]
a15r3 is a area. "An area (15) in r3".
the printed name of the a15r3 is "Room 3".
Understand "Area 15 in Room 3" as a15r3.
the parent of the a15r3 is r3.
the X of the a15r3 is "548181.300358915".
the Y of the a15r3 is "1133747.2423184358".

[create a18r3]
a18r3 is a area. "An area (18) in r3".
the printed name of the a18r3 is "Room 3".
Understand "Area 18 in Room 3" as a18r3.
the parent of the a18r3 is r3.
the X of the a18r3 is "547182.6594929246".
the Y of the a18r3 is "1134402.9197896435".

[create a19r3]
a19r3 is a area. "An area (19) in r3".
the printed name of the a19r3 is "Room 3".
Understand "Area 19 in Room 3" as a19r3.
the parent of the a19r3 is r3.
the X of the a19r3 is "547197.3683850407".
the Y of the a19r3 is "1134436.7470377567".

[create a46r3]
a46r3 is a area. "An area (46) in r3".
the printed name of the a46r3 is "Room 3".
Understand "Area 46 in Room 3" as a46r3.
the parent of the a46r3 is r3.
the X of the a46r3 is "547678.0".
the Y of the a46r3 is "1135132.0".

[create a47r3]
a47r3 is a area. "An area (47) in r3".
the printed name of the a47r3 is "Room 3".
Understand "Area 47 in Room 3" as a47r3.
the parent of the a47r3 is r3.
the X of the a47r3 is "547650.06582147".
the Y of the a47r3 is "1135081.2637312117".

[create a53r3]
a53r3 is a area. "An area (53) in r3".
the printed name of the a53r3 is "Room 3".
Understand "Area 53 in Room 3" as a53r3.
the parent of the a53r3 is r3.
the X of the a53r3 is "546889.0".
the Y of the a53r3 is "1134875.677419355".

[create a54r3]
a54r3 is a area. "An area (54) in r3".
the printed name of the a54r3 is "Room 3".
Understand "Area 54 in Room 3" as a54r3.
the parent of the a54r3 is r3.
the X of the a54r3 is "546911.3367070421".
the Y of the a54r3 is "1134839.012415756".

[create a64r3]
a64r3 is a area. "An area (64) in r3".
the printed name of the a64r3 is "Room 3".
Understand "Area 64 in Room 3" as a64r3.
the parent of the a64r3 is r3.
the X of the a64r3 is "547524.6354163943".
the Y of the a64r3 is "1134964.078181999".

[create a73r3]
a73r3 is a area. "An area (73) in r3".
the printed name of the a73r3 is "Room 3".
Understand "Area 73 in Room 3" as a73r3.
the parent of the a73r3 is r3.
the X of the a73r3 is "547289.5546762436".
the Y of the a73r3 is "1134406.593103916".

[create a88r3]
a88r3 is a area. "An area (88) in r3".
the printed name of the a88r3 is "Room 3".
Understand "Area 88 in Room 3" as a88r3.
the parent of the a88r3 is r3.
the X of the a88r3 is "547576.6766113773".
the Y of the a88r3 is "1134963.365108094".

[create a90r3]
a90r3 is a area. "An area (90) in r3".
the printed name of the a90r3 is "Room 3".
Understand "Area 90 in Room 3" as a90r3.
the parent of the a90r3 is r3.
the X of the a90r3 is "547068.5969734903".
the Y of the a90r3 is "1133851.1605624987".

[create a91r3]
a91r3 is a area. "An area (91) in r3".
the printed name of the a91r3 is "Room 3".
Understand "Area 91 in Room 3" as a91r3.
the parent of the a91r3 is r3.
the X of the a91r3 is "547994.7131916408".
the Y of the a91r3 is "1133856.5172281016".

[create a92r3]
a92r3 is a area. "An area (92) in r3".
the printed name of the a92r3 is "Room 3".
Understand "Area 92 in Room 3" as a92r3.
the parent of the a92r3 is r3.
the X of the a92r3 is "547108.7620334559".
the Y of the a92r3 is "1134194.2922717184".

[create a2r4]
a2r4 is a area. "An area (2) in r4".
the printed name of the a2r4 is "Room 4".
Understand "Area 2 in Room 4" as a2r4.
the parent of the a2r4 is r4.
the X of the a2r4 is "545098.0".
the Y of the a2r4 is "1132991.0".

[create a3r4]
a3r4 is a area. "An area (3) in r4".
the printed name of the a3r4 is "Room 4".
Understand "Area 3 in Room 4" as a3r4.
the parent of the a3r4 is r4.
the X of the a3r4 is "545129.2137161085".
the Y of the a3r4 is "1133016.633971292".

[create a12r4]
a12r4 is a area. "An area (12) in r4".
the printed name of the a12r4 is "Room 4".
Understand "Area 12 in Room 4" as a12r4.
the parent of the a12r4 is r4.
the X of the a12r4 is "546759.2520683949".
the Y of the a12r4 is "1133160.1036955323".

[create a13r4]
a13r4 is a area. "An area (13) in r4".
the printed name of the a13r4 is "Room 4".
Understand "Area 13 in Room 4" as a13r4.
the parent of the a13r4 is r4.
the X of the a13r4 is "546758.3899810578".
the Y of the a13r4 is "1133187.506819912".

[create a26r4]
a26r4 is a area. "An area (26) in r4".
the printed name of the a26r4 is "Room 4".
Understand "Area 26 in Room 4" as a26r4.
the parent of the a26r4 is r4.
the X of the a26r4 is "547291.0".
the Y of the a26r4 is "1132991.0".

[create a27r4]
a27r4 is a area. "An area (27) in r4".
the printed name of the a27r4 is "Room 4".
Understand "Area 27 in Room 4" as a27r4.
the parent of the a27r4 is r4.
the X of the a27r4 is "547259.7862838916".
the Y of the a27r4 is "1133016.633971292".

[create a30r4]
a30r4 is a area. "An area (30) in r4".
the printed name of the a30r4 is "Room 4".
Understand "Area 30 in Room 4" as a30r4.
the parent of the a30r4 is r4.
the X of the a30r4 is "545089.0".
the Y of the a30r4 is "1133080.0".

[create a31r4]
a31r4 is a area. "An area (31) in r4".
the printed name of the a31r4 is "Room 4".
Understand "Area 31 in Room 4" as a31r4.
the parent of the a31r4 is r4.
the X of the a31r4 is "545120.0059757696".
the Y of the a31r4 is "1133060.657907662".

[create a32r4]
a32r4 is a area. "An area (32) in r4".
the printed name of the a32r4 is "Room 4".
Understand "Area 32 in Room 4" as a32r4.
the parent of the a32r4 is r4.
the X of the a32r4 is "547300.0".
the Y of the a32r4 is "1133080.0".

[create a33r4]
a33r4 is a area. "An area (33) in r4".
the printed name of the a33r4 is "Room 4".
Understand "Area 33 in Room 4" as a33r4.
the parent of the a33r4 is r4.
the X of the a33r4 is "547259.7054722286".
the Y of the a33r4 is "1133061.5971994374".

[create a38r4]
a38r4 is a area. "An area (38) in r4".
the printed name of the a38r4 is "Room 4".
Understand "Area 38 in Room 4" as a38r4.
the parent of the a38r4 is r4.
the X of the a38r4 is "546531.0898876404".
the Y of the a38r4 is "1133000.0".

[create a39r4]
a39r4 is a area. "An area (39) in r4".
the printed name of the a39r4 is "Room 4".
Understand "Area 39 in Room 4" as a39r4.
the parent of the a39r4 is r4.
the X of the a39r4 is "546564.0417059886".
the Y of the a39r4 is "1132998.786189433".

[create a51r4]
a51r4 is a area. "An area (51) in r4".
the printed name of the a51r4 is "Room 4".
Understand "Area 51 in Room 4" as a51r4.
the parent of the a51r4 is r4.
the X of the a51r4 is "547212.0".
the Y of the a51r4 is "1133627.0".

[create a52r4]
a52r4 is a area. "An area (52) in r4".
the printed name of the a52r4 is "Room 4".
Understand "Area 52 in Room 4" as a52r4.
the parent of the a52r4 is r4.
the X of the a52r4 is "547148.0761500541".
the Y of the a52r4 is "1133656.0".

[create a68r4]
a68r4 is a area. "An area (68) in r4".
the printed name of the a68r4 is "Room 4".
Understand "Area 68 in Room 4" as a68r4.
the parent of the a68r4 is r4.
the X of the a68r4 is "546689.372443858".
the Y of the a68r4 is "1133254.4329165907".

[create a74r4]
a74r4 is a area. "An area (74) in r4".
the printed name of the a74r4 is "Room 4".
Understand "Area 74 in Room 4" as a74r4.
the parent of the a74r4 is r4.
the X of the a74r4 is "547243.6659021408".
the Y of the a74r4 is "1133043.50127421".

[create a76r4]
a76r4 is a area. "An area (76) in r4".
the printed name of the a76r4 is "Room 4".
Understand "Area 76 in Room 4" as a76r4.
the parent of the a76r4 is r4.
the X of the a76r4 is "545143.389199255".
the Y of the a76r4 is "1133040.2597765364".

[create a78r4]
a78r4 is a area. "An area (78) in r4".
the printed name of the a78r4 is "Room 4".
Understand "Area 78 in Room 4" as a78r4.
the parent of the a78r4 is r4.
the X of the a78r4 is "546576.8467147517".
the Y of the a78r4 is "1133138.7380611144".

[create a80r4]
a80r4 is a area. "An area (80) in r4".
the printed name of the a80r4 is "Room 4".
Understand "Area 80 in Room 4" as a80r4.
the parent of the a80r4 is r4.
the X of the a80r4 is "546606.1433255486".
the Y of the a80r4 is "1133147.3467319268".

[create a81r4]
a81r4 is a area. "An area (81) in r4".
the printed name of the a81r4 is "Room 4".
Understand "Area 81 in Room 4" as a81r4.
the parent of the a81r4 is r4.
the X of the a81r4 is "546906.3961068327".
the Y of the a81r4 is "1133128.733816152".

[create a83r4]
a83r4 is a area. "An area (83) in r4".
the printed name of the a83r4 is "Room 4".
Understand "Area 83 in Room 4" as a83r4.
the parent of the a83r4 is r4.
the X of the a83r4 is "546883.1857401195".
the Y of the a83r4 is "1133357.1279414608".

[create a85r4]
a85r4 is a area. "An area (85) in r4".
the printed name of the a85r4 is "Room 4".
Understand "Area 85 in Room 4" as a85r4.
the parent of the a85r4 is r4.
the X of the a85r4 is "545527.1865194305".
the Y of the a85r4 is "1133170.3208574706".

[create a89r4]
a89r4 is a area. "An area (89) in r4".
the printed name of the a89r4 is "Room 4".
Understand "Area 89 in Room 4" as a89r4.
the parent of the a89r4 is r4.
the X of the a89r4 is "547027.2515729473".
the Y of the a89r4 is "1133701.429490181".

[create a4r5]
a4r5 is a area. "An area (4) in r5".
the printed name of the a4r5 is "Room 5".
Understand "Area 4 in Room 5" as a4r5.
the parent of the a4r5 is r5.
the X of the a4r5 is "545111.0".
the Y of the a4r5 is "1135024.0".

[create a5r5]
a5r5 is a area. "An area (5) in r5".
the printed name of the a5r5 is "Room 5".
Understand "Area 5 in Room 5" as a5r5.
the parent of the a5r5 is r5.
the X of the a5r5 is "545153.06".
the Y of the a5r5 is "1135042.519".

[create a8r5]
a8r5 is a area. "An area (8) in r5".
the printed name of the a8r5 is "Room 5".
Understand "Area 8 in Room 5" as a8r5.
the parent of the a8r5 is r5.
the X of the a8r5 is "545120.0".
the Y of the a8r5 is "1135114.0".

[create a9r5]
a9r5 is a area. "An area (9) in r5".
the printed name of the a9r5 is "Room 5".
Understand "Area 9 in Room 5" as a9r5.
the parent of the a9r5 is r5.
the X of the a9r5 is "545150.9153780071".
the Y of the a9r5 is "1135088.1834621993".

[create a44r5]
a44r5 is a area. "An area (44) in r5".
the printed name of the a44r5 is "Room 5".
Understand "Area 44 in Room 5" as a44r5.
the parent of the a44r5 is r5.
the X of the a44r5 is "546695.8043478262".
the Y of the a44r5 is "1134975.0".

[create a45r5]
a45r5 is a area. "An area (45) in r5".
the printed name of the a45r5 is "Room 5".
Understand "Area 45 in Room 5" as a45r5.
the parent of the a45r5 is r5.
the X of the a45r5 is "546659.1632790826".
the Y of the a45r5 is "1134983.566457224".

[create a63r5]
a63r5 is a area. "An area (63) in r5".
the printed name of the a63r5 is "Room 5".
Understand "Area 63 in Room 5" as a63r5.
the parent of the a63r5 is r5.
the X of the a63r5 is "545168.092671901".
the Y of the a63r5 is "1135059.8065726862".

[create a86r5]
a86r5 is a area. "An area (86) in r5".
the printed name of the a86r5 is "Room 5".
Understand "Area 86 in Room 5" as a86r5.
the parent of the a86r5 is r5.
the X of the a86r5 is "545527.6980415579".
the Y of the a86r5 is "1134933.2713725318".

[create a24r6]
a24r6 is a area. "An area (24) in r6".
the printed name of the a24r6 is "Room 6".
Understand "Area 24 in Room 6" as a24r6.
the parent of the a24r6 is r6.
the X of the a24r6 is "546889.0".
the Y of the a24r6 is "1134975.0".

[create a25r6]
a25r6 is a area. "An area (25) in r6".
the printed name of the a25r6 is "Room 6".
Understand "Area 25 in Room 6" as a25r6.
the parent of the a25r6 is r6.
the X of the a25r6 is "546856.8007246377".
the Y of the a25r6 is "1134950.1693548388".

[create a42r6]
a42r6 is a area. "An area (42) in r6".
the printed name of the a42r6 is "Room 6".
Understand "Area 42 in Room 6" as a42r6.
the parent of the a42r6 is r6.
the X of the a42r6 is "546714.0".
the Y of the a42r6 is "1134966.0".

[create a43r6]
a43r6 is a area. "An area (43) in r6".
the printed name of the a43r6 is "Room 6".
Understand "Area 43 in Room 6" as a43r6.
the parent of the a43r6 is r6.
the X of the a43r6 is "546792.4021739131".
the Y of the a43r6 is "1134963.552540156".

[create a44r6]
a44r6 is a area. "An area (44) in r6".
the printed name of the a44r6 is "Room 6".
Understand "Area 44 in Room 6" as a44r6.
the parent of the a44r6 is r6.
the X of the a44r6 is "546695.8043478262".
the Y of the a44r6 is "1134975.0".

[create a53r6]
a53r6 is a area. "An area (53) in r6".
the printed name of the a53r6 is "Room 6".
Understand "Area 53 in Room 6" as a53r6.
the parent of the a53r6 is r6.
the X of the a53r6 is "546889.0".
the Y of the a53r6 is "1134875.677419355".

[create a59r6]
a59r6 is a area. "An area (59) in r6".
the printed name of the a59r6 is "Room 6".
Understand "Area 59 in Room 6" as a59r6.
the parent of the a59r6 is r6.
the X of the a59r6 is "546873.7080715646".
the Y of the a59r6 is "1134900.7786399627".

[create a62r6]
a62r6 is a area. "An area (62) in r6".
the printed name of the a62r6 is "Room 6".
Understand "Area 62 in Room 6" as a62r6.
the parent of the a62r6 is r6.
the X of the a62r6 is "546850.8286071356".
the Y of the a62r6 is "1134942.4249786977".

[create a55r7]
a55r7 is a area. "An area (55) in r7".
the printed name of the a55r7 is "Room 7".
Understand "Area 55 in Room 7" as a55r7.
the parent of the a55r7 is r7.
the X of the a55r7 is "549963.8444444444".
the Y of the a55r7 is "1134997.0".

[create a57r7]
a57r7 is a area. "An area (57) in r7".
the printed name of the a57r7 is "Room 7".
Understand "Area 57 in Room 7" as a57r7.
the parent of the a57r7 is r7.
the X of the a57r7 is "549997.0".
the Y of the a57r7 is "1134960.31147541".

[create d0]
d0 is a door. "A door between a14r1 and a15r3".
d0 is northwest of a14r1 and southeast of a15r3.

[create d1]
d1 is a door. "A door between a71r1 and a15r3".
d1 is east of a71r1 and west of a15r3.

[create d2]
d2 is a door. "A door between a85r4 and a86r5".
d2 is north of a85r4 and south of a86r5.

[create d4]
d4 is a door. "A door between a16r1 and a17r2".
d4 is west of a16r1 and east of a17r2.

[create d5]
d5 is a door. "A door between a67r1 and a72r2".
d5 is northeast of a67r1 and southwest of a72r2.

[create d6]
d6 is a door. "A door between a60r0 and a70r1".
d6 is south of a60r0 and north of a70r1.

[create d8]
d8 is a door. "A door between a71r1 and a91r3".
d8 is northwest of a71r1 and southeast of a91r3.

[create d10]
d10 is a door. "A door between a90r3 and a89r4".
d10 is south of a90r3 and north of a89r4.

north of a0r1 is south of a1r1.

northeast of a2r4 is southwest of a3r4.

northeast of a4r5 is southwest of a5r5.

southeast of a6r3 is northwest of a7r3.

southeast of a8r5 is northwest of a9r5.

northwest of a10r1 is southeast of a11r1.

north of a12r4 is south of a13r4.

northeast of a18r3 is southwest of a19r3.

east of a20r0 is west of a21r0.

southwest of a22r2 is northeast of a23r2.

southwest of a24r6 is northeast of a25r6.

northwest of a26r4 is southeast of a27r4.

northeast of a28r1 is southwest of a29r1.

southeast of a30r4 is northwest of a31r4.

southwest of a32r4 is northeast of a33r4.

southeast of a34r1 is northwest of a35r1.

west of a36r1 is east of a37r1.

east of a38r4 is west of a39r4.

northwest of a40r1 is southeast of a41r1.

east of a42r6 is west of a43r6.

west of a44r5 is east of a45r5.

southwest of a46r3 is northeast of a47r3.

southwest of a48r1 is northeast of a49r1.

east of a36r1 is west of a50r1.

northwest of a51r4 is southeast of a52r4.

southeast of a53r3 is northwest of a54r3.

northwest of a55r0 is southeast of a56r0.

south of a57r2 is north of a58r2.

east of a53r3 is west of a53r3.

east of a21r0 is west of a60r0.

west of a23r2 is east of a61r2.

southwest of a25r6 is northeast of a62r6.

northeast of a5r5 is southwest of a63r5.

southeast of a7r3 is northwest of a64r3.

southeast of a9r5 is northwest of a63r5.

northeast of a41r1 is southwest of a65r1.

southeast of a65r1 is northwest of a66r1.

northwest of a11r1 is southeast of a67r1.

northwest of a13r4 is southeast of a68r4.

east of a69r1 is west of a70r1.

west of a17r2 is east of a72r2.

east of a19r3 is west of a73r3.

northwest of a27r4 is southeast of a74r4.

northeast of a29r1 is southwest of a75r1.

southeast of a31r4 is northwest of a76r4.

southwest of a33r4 is northeast of a74r4.

southeast of a35r1 is northwest of a75r1.

north of a37r1 is south of a77r1.

southwest of a66r1 is northeast of a77r1.

north of a39r4 is south of a78r4.

north of a1r1 is south of a79r1.

northeast of a3r4 is southwest of a76r4.

east of a80r4 is west of a81r4.

southeast of a66r1 is northwest of a82r1.

north of a81r4 is south of a83r4.

east of a78r4 is west of a80r4.

south of a69r1 is north of a84r1.

northwest of a65r1 is southeast of a70r1.

southwest of a68r4 is northeast of a80r4.

east of a75r1 is west of a84r1.

east of a76r4 is west of a85r4.

east of a82r1 is west of a87r1.

west of a74r4 is east of a81r4.

west of a77r1 is east of a84r1.

northeast of a50r1 is southwest of a82r1.

southwest of a67r1 is northeast of a87r1.

west of a78r4 is east of a85r4.

west of a49r1 is east of a79r1.

northwest of a79r1 is southeast of a87r1.

south of a58r2 is north of a61r2.

northwest of a59r6 is southeast of a62r6.

east of a43r6 is west of a62r6.

west of a45r5 is east of a86r5.

east of a63r5 is west of a86r5.

east of a64r3 is west of a88r3.

southwest of a47r3 is northeast of a88r3.

south of a61r2 is north of a72r2.

south of a88r3 is north of a91r3.

north of a90r3 is south of a92r3.

southwest of a73r3 is northeast of a92r3.

south of a54r3 is north of a92r3.

southwest of a64r3 is northeast of a73r3.

west of a56r0 is east of a60r0.

northeast of a68r4 is southwest of a83r4.

northeast of a83r4 is southwest of a89r4.

northwest of a69r1 is southeast of a71r1.

west of a52r4 is east of a89r4.

east of a90r3 is west of a91r3.

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
the player is in a62r6.

the orientation of the player is 6.
