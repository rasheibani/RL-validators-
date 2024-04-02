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

[create a0r0]
a0r0 is a area. "An area (0) in r0".
the printed name of the a0r0 is "Room 0".
Understand "Area 0 in Room 0" as a0r0.
the parent of the a0r0 is r0.
the X of the a0r0 is "548835.0".
the Y of the a0r0 is "1013514.0".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "548865.8949627932".
the Y of the a1r0 is "1013487.7894485786".

[create a2r0]
a2r0 is a area. "An area (2) in r0".
the printed name of the a2r0 is "Room 0".
Understand "Area 2 in Room 0" as a2r0.
the parent of the a2r0 is r0.
the X of the a2r0 is "549593.0".
the Y of the a2r0 is "1013514.0".

[create a3r0]
a3r0 is a area. "An area (3) in r0".
the printed name of the a3r0 is "Room 0".
Understand "Area 3 in Room 0" as a3r0.
the parent of the a3r0 is r0.
the X of the a3r0 is "549562.1050372068".
the Y of the a3r0 is "1013487.7894485785".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "548825.0".
the Y of the a22r0 is "1013424.0".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "548863.7029286582".
the Y of the a23r0 is "1013442.4774523712".

[create a24r0]
a24r0 is a area. "An area (24) in r0".
the printed name of the a24r0 is "Room 0".
Understand "Area 24 in Room 0" as a24r0.
the parent of the a24r0 is r0.
the X of the a24r0 is "549603.0".
the Y of the a24r0 is "1013424.0".

[create a25r0]
a25r0 is a area. "An area (25) in r0".
the printed name of the a25r0 is "Room 0".
Understand "Area 25 in Room 0" as a25r0.
the parent of the a25r0 is r0.
the X of the a25r0 is "549563.9982014389".
the Y of the a25r0 is "1013442.4442446043".

[create a69r0]
a69r0 is a area. "An area (69) in r0".
the printed name of the a69r0 is "Room 0".
Understand "Area 69 in Room 0" as a69r0.
the parent of the a69r0 is r0.
the X of the a69r0 is "548881.7072666291".
the Y of the a69r0 is "1013461.4777749957".

[create a70r0]
a70r0 is a area. "An area (70) in r0".
the printed name of the a70r0 is "Room 0".
Understand "Area 70 in Room 0" as a70r0.
the parent of the a70r0 is r0.
the X of the a70r0 is "549546.3051002843".
the Y of the a70r0 is "1013461.49835354".

[create a77r0]
a77r0 is a area. "An area (77) in r0".
the printed name of the a77r0 is "Room 0".
Understand "Area 77 in Room 0" as a77r0.
the parent of the a77r0 is r0.
the X of the a77r0 is "549225.3793965107".
the Y of the a77r0 is "1013374.0644405335".

[create a12r1]
a12r1 is a area. "An area (12) in r1".
the printed name of the a12r1 is "Room 1".
Understand "Area 12 in Room 1" as a12r1.
the parent of the a12r1 is r1.
the X of the a12r1 is "547924.0".
the Y of the a12r1 is "1012125.0".

[create a13r1]
a13r1 is a area. "An area (13) in r1".
the printed name of the a13r1 is "Room 1".
Understand "Area 13 in Room 1" as a13r1.
the parent of the a13r1 is r1.
the X of the a13r1 is "547892.8077975204".
the Y of the a13r1 is "1012151.4338578527".

[create a28r1]
a28r1 is a area. "An area (28) in r1".
the printed name of the a28r1 is "Room 1".
Understand "Area 28 in Room 1" as a28r1.
the parent of the a28r1 is r1.
the X of the a28r1 is "545777.0".
the Y of the a28r1 is "1011480.0".

[create a29r1]
a29r1 is a area. "An area (29) in r1".
the printed name of the a29r1 is "Room 1".
Understand "Area 29 in Room 1" as a29r1.
the parent of the a29r1 is r1.
the X of the a29r1 is "545736.7054722288".
the Y of the a29r1 is "1011461.5971994374".

[create a30r1]
a30r1 is a area. "An area (30) in r1".
the printed name of the a30r1 is "Room 1".
Understand "Area 30 in Room 1" as a30r1.
the parent of the a30r1 is r1.
the X of the a30r1 is "546272.0".
the Y of the a30r1 is "1011480.0".

[create a31r1]
a31r1 is a area. "An area (31) in r1".
the printed name of the a31r1 is "Room 1".
Understand "Area 31 in Room 1" as a31r1.
the parent of the a31r1 is r1.
the X of the a31r1 is "546313.3556858185".
the Y of the a31r1 is "1011461.7045075545".

[create a32r1]
a32r1 is a area. "An area (32) in r1".
the printed name of the a32r1 is "Room 1".
Understand "Area 32 in Room 1" as a32r1.
the parent of the a32r1 is r1.
the X of the a32r1 is "549019.0".
the Y of the a32r1 is "1011480.0".

[create a33r1]
a33r1 is a area. "An area (33) in r1".
the printed name of the a33r1 is "Room 1".
Understand "Area 33 in Room 1" as a33r1.
the parent of the a33r1 is r1.
the X of the a33r1 is "548975.9383292026".
the Y of the a33r1 is "1011461.8770228897".

[create a34r1]
a34r1 is a area. "An area (34) in r1".
the printed name of the a34r1 is "Room 1".
Understand "Area 34 in Room 1" as a34r1.
the parent of the a34r1 is r1.
the X of the a34r1 is "546623.0898876404".
the Y of the a34r1 is "1011400.0".

[create a35r1]
a35r1 is a area. "An area (35) in r1".
the printed name of the a35r1 is "Room 1".
Understand "Area 35 in Room 1" as a35r1.
the parent of the a35r1 is r1.
the X of the a35r1 is "546656.0417059886".
the Y of the a35r1 is "1011398.7861894328".

[create a36r1]
a36r1 is a area. "An area (36) in r1".
the printed name of the a36r1 is "Room 1".
Understand "Area 36 in Room 1" as a36r1.
the parent of the a36r1 is r1.
the X of the a36r1 is "547123.9101123596".
the Y of the a36r1 is "1011400.0".

[create a37r1]
a37r1 is a area. "An area (37) in r1".
the printed name of the a37r1 is "Room 1".
Understand "Area 37 in Room 1" as a37r1.
the parent of the a37r1 is r1.
the X of the a37r1 is "547090.9582940115".
the Y of the a37r1 is "1011398.7861894328".

[create a38r1]
a38r1 is a area. "An area (38) in r1".
the printed name of the a38r1 is "Room 1".
Understand "Area 38 in Room 1" as a38r1.
the parent of the a38r1 is r1.
the X of the a38r1 is "548295.0".
the Y of the a38r1 is "1011888.0".

[create a39r1]
a39r1 is a area. "An area (39) in r1".
the printed name of the a39r1 is "Room 1".
Understand "Area 39 in Room 1" as a39r1.
the parent of the a39r1 is r1.
the X of the a39r1 is "548261.151112356".
the Y of the a39r1 is "1011867.8660087992".

[create a40r1]
a40r1 is a area. "An area (40) in r1".
the printed name of the a40r1 is "Room 1".
Understand "Area 40 in Room 1" as a40r1.
the parent of the a40r1 is r1.
the X of the a40r1 is "548212.0".
the Y of the a40r1 is "1011925.0".

[create a41r1]
a41r1 is a area. "An area (41) in r1".
the printed name of the a41r1 is "Room 1".
Understand "Area 41 in Room 1" as a41r1.
the parent of the a41r1 is r1.
the X of the a41r1 is "548218.5365348752".
the Y of the a41r1 is "1011883.8657403952".

[create a42r1]
a42r1 is a area. "An area (42) in r1".
the printed name of the a42r1 is "Room 1".
Understand "Area 42 in Room 1" as a42r1.
the parent of the a42r1 is r1.
the X of the a42r1 is "547816.0".
the Y of the a42r1 is "1012110.0".

[create a43r1]
a43r1 is a area. "An area (43) in r1".
the printed name of the a43r1 is "Room 1".
Understand "Area 43 in Room 1" as a43r1.
the parent of the a43r1 is r1.
the X of the a43r1 is "547839.0755088732".
the Y of the a43r1 is "1012142.0063361116".

[create a44r1]
a44r1 is a area. "An area (44) in r1".
the printed name of the a44r1 is "Room 1".
Understand "Area 44 in Room 1" as a44r1.
the parent of the a44r1 is r1.
the X of the a44r1 is "545009.0".
the Y of the a44r1 is "1011391.0".

[create a45r1]
a45r1 is a area. "An area (45) in r1".
the printed name of the a45r1 is "Room 1".
Understand "Area 45 in Room 1" as a45r1.
the parent of the a45r1 is r1.
the X of the a45r1 is "545040.2137161086".
the Y of the a45r1 is "1011416.633971292".

[create a46r1]
a46r1 is a area. "An area (46) in r1".
the printed name of the a46r1 is "Room 1".
Understand "Area 46 in Room 1" as a46r1.
the parent of the a46r1 is r1.
the X of the a46r1 is "545768.0".
the Y of the a46r1 is "1011391.0".

[create a47r1]
a47r1 is a area. "An area (47) in r1".
the printed name of the a47r1 is "Room 1".
Understand "Area 47 in Room 1" as a47r1.
the parent of the a47r1 is r1.
the X of the a47r1 is "545736.7862838916".
the Y of the a47r1 is "1011416.6339712921".

[create a48r1]
a48r1 is a area. "An area (48) in r1".
the printed name of the a48r1 is "Room 1".
Understand "Area 48 in Room 1" as a48r1.
the parent of the a48r1 is r1.
the X of the a48r1 is "546281.0".
the Y of the a48r1 is "1011391.0".

[create a49r1]
a49r1 is a area. "An area (49) in r1".
the printed name of the a49r1 is "Room 1".
Understand "Area 49 in Room 1" as a49r1.
the parent of the a49r1 is r1.
the X of the a49r1 is "546312.2137161085".
the Y of the a49r1 is "1011416.633971292".

[create a50r1]
a50r1 is a area. "An area (50) in r1".
the printed name of the a50r1 is "Room 1".
Understand "Area 50 in Room 1" as a50r1.
the parent of the a50r1 is r1.
the X of the a50r1 is "548167.0898876404".
the Y of the a50r1 is "1011400.0".

[create a51r1]
a51r1 is a area. "An area (51) in r1".
the printed name of the a51r1 is "Room 1".
Understand "Area 51 in Room 1" as a51r1.
the parent of the a51r1 is r1.
the X of the a51r1 is "548200.0417059886".
the Y of the a51r1 is "1011398.7861894328".

[create a52r1]
a52r1 is a area. "An area (52) in r1".
the printed name of the a52r1 is "Room 1".
Understand "Area 52 in Room 1" as a52r1.
the parent of the a52r1 is r1.
the X of the a52r1 is "549010.0".
the Y of the a52r1 is "1011391.0".

[create a53r1]
a53r1 is a area. "An area (53) in r1".
the printed name of the a53r1 is "Room 1".
Understand "Area 53 in Room 1" as a53r1.
the parent of the a53r1 is r1.
the X of the a53r1 is "548978.7862838916".
the Y of the a53r1 is "1011416.6339712921".

[create a54r1]
a54r1 is a area. "An area (54) in r1".
the printed name of the a54r1 is "Room 1".
Understand "Area 54 in Room 1" as a54r1.
the parent of the a54r1 is r1.
the X of the a54r1 is "545000.0".
the Y of the a54r1 is "1011480.0".

[create a55r1]
a55r1 is a area. "An area (55) in r1".
the printed name of the a55r1 is "Room 1".
Understand "Area 55 in Room 1" as a55r1.
the parent of the a55r1 is r1.
the X of the a55r1 is "545040.2945277713".
the Y of the a55r1 is "1011461.5971994374".

[create a58r1]
a58r1 is a area. "An area (58) in r1".
the printed name of the a58r1 is "Room 1".
Understand "Area 58 in Room 1" as a58r1.
the parent of the a58r1 is r1.
the X of the a58r1 is "548961.7290448344".
the Y of the a58r1 is "1011445.0627030539".

[create a59r1]
a59r1 is a area. "An area (59) in r1".
the printed name of the a59r1 is "Room 1".
Understand "Area 59 in Room 1" as a59r1.
the parent of the a59r1 is r1.
the X of the a59r1 is "546667.7719734608".
the Y of the a59r1 is "1011526.9917157571".

[create a60r1]
a60r1 is a area. "An area (60) in r1".
the printed name of the a60r1 is "Room 1".
Understand "Area 60 in Room 1" as a60r1.
the parent of the a60r1 is r1.
the X of the a60r1 is "547076.9037356027".
the Y of the a60r1 is "1011552.394967946".

[create a61r1]
a61r1 is a area. "An area (61) in r1".
the printed name of the a61r1 is "Room 1".
Understand "Area 61 in Room 1" as a61r1.
the parent of the a61r1 is r1.
the X of the a61r1 is "546710.1204613352".
the Y of the a61r1 is "1011561.3072492096".

[create a62r1]
a62r1 is a area. "An area (62) in r1".
the printed name of the a62r1 is "Room 1".
Understand "Area 62 in Room 1" as a62r1.
the parent of the a62r1 is r1.
the X of the a62r1 is "547057.420638165".
the Y of the a62r1 is "1011567.5887633123".

[create a63r1]
a63r1 is a area. "An area (63) in r1".
the printed name of the a63r1 is "Room 1".
Understand "Area 63 in Room 1" as a63r1.
the parent of the a63r1 is r1.
the X of the a63r1 is "548237.3706921167".
the Y of the a63r1 is "1011860.5041415091".

[create a64r1]
a64r1 is a area. "An area (64) in r1".
the printed name of the a64r1 is "Room 1".
Understand "Area 64 in Room 1" as a64r1.
the parent of the a64r1 is r1.
the X of the a64r1 is "547860.9241359931".
the Y of the a64r1 is "1012172.8985341636".

[create a65r1]
a65r1 is a area. "An area (65) in r1".
the printed name of the a65r1 is "Room 1".
Understand "Area 65 in Room 1" as a65r1.
the parent of the a65r1 is r1.
the X of the a65r1 is "545056.3340978592".
the Y of the a65r1 is "1011443.5012742098".

[create a66r1]
a66r1 is a area. "An area (66) in r1".
the printed name of the a66r1 is "Room 1".
Understand "Area 66 in Room 1" as a66r1.
the parent of the a66r1 is r1.
the X of the a66r1 is "545720.6659021407".
the Y of the a66r1 is "1011443.5012742099".

[create a67r1]
a67r1 is a area. "An area (67) in r1".
the printed name of the a67r1 is "Room 1".
Understand "Area 67 in Room 1" as a67r1.
the parent of the a67r1 is r1.
the X of the a67r1 is "546328.634469697".
the Y of the a67r1 is "1011444.0018939393".

[create a68r1]
a68r1 is a area. "An area (68) in r1".
the printed name of the a68r1 is "Room 1".
Understand "Area 68 in Room 1" as a68r1.
the parent of the a68r1 is r1.
the X of the a68r1 is "548204.0304135778".
the Y of the a68r1 is "1011442.3806221308".

[create a83r1]
a83r1 is a area. "An area (83) in r1".
the printed name of the a83r1 is "Room 1".
Understand "Area 83 in Room 1" as a83r1.
the parent of the a83r1 is r1.
the X of the a83r1 is "546472.9076679603".
the Y of the a83r1 is "1012256.9294065498".

[create a84r1]
a84r1 is a area. "An area (84) in r1".
the printed name of the a84r1 is "Room 1".
Understand "Area 84 in Room 1" as a84r1.
the parent of the a84r1 is r1.
the X of the a84r1 is "548590.1730804583".
the Y of the a84r1 is "1012259.7005024817".

[create a85r1]
a85r1 is a area. "An area (85) in r1".
the printed name of the a85r1 is "Room 1".
Understand "Area 85 in Room 1" as a85r1.
the parent of the a85r1 is r1.
the X of the a85r1 is "548595.0".
the Y of the a85r1 is "1011568.5778301886".

[create a86r1]
a86r1 is a area. "An area (86) in r1".
the printed name of the a86r1 is "Room 1".
Understand "Area 86 in Room 1" as a86r1.
the parent of the a86r1 is r1.
the X of the a86r1 is "545604.9376088154".
the Y of the a86r1 is "1012242.9379071898".

[create a87r1]
a87r1 is a area. "An area (87) in r1".
the printed name of the a87r1 is "Room 1".
Understand "Area 87 in Room 1" as a87r1.
the parent of the a87r1 is r1.
the X of the a87r1 is "545383.3961068324".
the Y of the a87r1 is "1011528.7338161519".

[create a88r1]
a88r1 is a area. "An area (88) in r1".
the printed name of the a88r1 is "Room 1".
Understand "Area 88 in Room 1" as a88r1.
the parent of the a88r1 is r1.
the X of the a88r1 is "548112.611290454".
the Y of the a88r1 is "1011514.9026993677".

[create a26r2]
a26r2 is a area. "An area (26) in r2".
the printed name of the a26r2 is "Room 2".
Understand "Area 26 in Room 2" as a26r2.
the parent of the a26r2 is r2.
the X of the a26r2 is "545888.0".
the Y of the a26r2 is "1013501.0".

[create a27r2]
a27r2 is a area. "An area (27) in r2".
the printed name of the a27r2 is "Room 2".
Understand "Area 27 in Room 2" as a27r2.
the parent of the a27r2 is r2.
the X of the a27r2 is "545928.1595627209".
the Y of the a27r2 is "1013446.4402178592".

[create a56r2]
a56r2 is a area. "An area (56) in r2".
the printed name of the a56r2 is "Room 2".
Understand "Area 56 in Room 2" as a56r2.
the parent of the a56r2 is r2.
the X of the a56r2 is "546155.0".
the Y of the a56r2 is "1013532.0".

[create a57r2]
a57r2 is a area. "An area (57) in r2".
the printed name of the a57r2 is "Room 2".
Understand "Area 57 in Room 2" as a57r2.
the parent of the a57r2 is r2.
the X of the a57r2 is "546127.259315724".
the Y of the a57r2 is "1013479.5971839246".

[create a76r2]
a76r2 is a area. "An area (76) in r2".
the printed name of the a76r2 is "Room 2".
Understand "Area 76 in Room 2" as a76r2.
the parent of the a76r2 is r2.
the X of the a76r2 is "546002.5871593204".
the Y of the a76r2 is "1013364.2265159851".

[create a78r2]
a78r2 is a area. "An area (78) in r2".
the printed name of the a78r2 is "Room 2".
Understand "Area 78 in Room 2" as a78r2.
the parent of the a78r2 is r2.
the X of the a78r2 is "546054.7557274038".
the Y of the a78r2 is "1013363.3385288817".

[create a4r3]
a4r3 is a area. "An area (4) in r3".
the printed name of the a4r3 is "Room 3".
Understand "Area 4 in Room 3" as a4r3.
the parent of the a4r3 is r3.
the X of the a4r3 is "546649.0".
the Y of the a4r3 is "1013514.0".

[create a5r3]
a5r3 is a area. "An area (5) in r3".
the printed name of the a5r3 is "Room 3".
Understand "Area 5 in Room 3" as a5r3.
the parent of the a5r3 is r3.
the X of the a5r3 is "546679.9153780069".
the Y of the a5r3 is "1013488.1834621993".

[create a6r3]
a6r3 is a area. "An area (6) in r3".
the printed name of the a6r3 is "Room 3".
Understand "Area 6 in Room 3" as a6r3.
the parent of the a6r3 is r3.
the X of the a6r3 is "548451.0".
the Y of the a6r3 is "1013514.0".

[create a7r3]
a7r3 is a area. "An area (7) in r3".
the printed name of the a7r3 is "Room 3".
Understand "Area 7 in Room 3" as a7r3.
the parent of the a7r3 is r3.
the X of the a7r3 is "548419.9123931624".
the Y of the a7r3 is "1013488.1662393163".

[create a8r3]
a8r3 is a area. "An area (8) in r3".
the printed name of the a8r3 is "Room 3".
Understand "Area 8 in Room 3" as a8r3.
the parent of the a8r3 is r3.
the X of the a8r3 is "547609.0".
the Y of the a8r3 is "1013514.0".

[create a9r3]
a9r3 is a area. "An area (9) in r3".
the printed name of the a9r3 is "Room 3".
Understand "Area 9 in Room 3" as a9r3.
the parent of the a9r3 is r3.
the X of the a9r3 is "547640.826460481".
the Y of the a9r3 is "1013505.7673539519".

[create a10r3]
a10r3 is a area. "An area (10) in r3".
the printed name of the a10r3 is "Room 3".
Understand "Area 10 in Room 3" as a10r3.
the parent of the a10r3 is r3.
the X of the a10r3 is "548163.0".
the Y of the a10r3 is "1013523.0".

[create a11r3]
a11r3 is a area. "An area (11) in r3".
the printed name of the a11r3 is "Room 3".
Understand "Area 11 in Room 3" as a11r3.
the parent of the a11r3 is r3.
the X of the a11r3 is "548132.037202381".
the Y of the a11r3 is "1013513.5".

[create a14r3]
a14r3 is a area. "An area (14) in r3".
the printed name of the a14r3 is "Room 3".
Understand "Area 14 in Room 3" as a14r3.
the parent of the a14r3 is r3.
the X of the a14r3 is "547924.0".
the Y of the a14r3 is "1012810.0".

[create a15r3]
a15r3 is a area. "An area (15) in r3".
the printed name of the a15r3 is "Room 3".
Understand "Area 15 in Room 3" as a15r3.
the parent of the a15r3 is r3.
the X of the a15r3 is "547893.7769924195".
the Y of the a15r3 is "1012783.9929317763".

[create a16r3]
a16r3 is a area. "An area (16) in r3".
the printed name of the a16r3 is "Room 3".
Understand "Area 16 in Room 3" as a16r3.
the parent of the a16r3 is r3.
the X of the a16r3 is "547816.0".
the Y of the a16r3 is "1012822.0".

[create a17r3]
a17r3 is a area. "An area (17) in r3".
the printed name of the a17r3 is "Room 3".
Understand "Area 17 in Room 3" as a17r3.
the parent of the a17r3 is r3.
the X of the a17r3 is "547839.7621979024".
the Y of the a17r3 is "1012789.8597811217".

[create a18r3]
a18r3 is a area. "An area (18) in r3".
the printed name of the a18r3 is "Room 3".
Understand "Area 18 in Room 3" as a18r3.
the parent of the a18r3 is r3.
the X of the a18r3 is "546640.0".
the Y of the a18r3 is "1013424.0".

[create a19r3]
a19r3 is a area. "An area (19) in r3".
the printed name of the a19r3 is "Room 3".
Understand "Area 19 in Room 3" as a19r3.
the parent of the a19r3 is r3.
the X of the a19r3 is "546682.06".
the Y of the a19r3 is "1013442.519".

[create a20r3]
a20r3 is a area. "An area (20) in r3".
the printed name of the a20r3 is "Room 3".
Understand "Area 20 in Room 3" as a20r3.
the parent of the a20r3 is r3.
the X of the a20r3 is "548460.0".
the Y of the a20r3 is "1013424.0".

[create a21r3]
a21r3 is a area. "An area (21) in r3".
the printed name of the a21r3 is "Room 3".
Understand "Area 21 in Room 3" as a21r3.
the parent of the a21r3 is r3.
the X of the a21r3 is "548420.9448424069".
the Y of the a21r3 is "1013442.8194842407".

[create a71r3]
a71r3 is a area. "An area (71) in r3".
the printed name of the a71r3 is "Room 3".
Understand "Area 71 in Room 3" as a71r3.
the parent of the a71r3 is r3.
the X of the a71r3 is "546697.092671901".
the Y of the a71r3 is "1013459.8065726861".

[create a72r3]
a72r3 is a area. "An area (72) in r3".
the printed name of the a72r3 is "Room 3".
Understand "Area 72 in Room 3" as a72r3.
the parent of the a72r3 is r3.
the X of the a72r3 is "548403.915169287".
the Y of the a72r3 is "1013461.6108476832".

[create a73r3]
a73r3 is a area. "An area (73) in r3".
the printed name of the a73r3 is "Room 3".
Understand "Area 73 in Room 3" as a73r3.
the parent of the a73r3 is r3.
the X of the a73r3 is "547646.9973669563".
the Y of the a73r3 is "1013445.2924704915".

[create a74r3]
a74r3 is a area. "An area (74) in r3".
the printed name of the a74r3 is "Room 3".
Understand "Area 74 in Room 3" as a74r3.
the parent of the a74r3 is r3.
the X of the a74r3 is "548112.5169467654".
the Y of the a74r3 is "1013427.1999225424".

[create a75r3]
a75r3 is a area. "An area (75) in r3".
the printed name of the a75r3 is "Room 3".
Understand "Area 75 in Room 3" as a75r3.
the parent of the a75r3 is r3.
the X of the a75r3 is "547860.858357395".
the Y of the a75r3 is "1012760.9834777031".

[create a79r3]
a79r3 is a area. "An area (79) in r3".
the printed name of the a79r3 is "Room 3".
Understand "Area 79 in Room 3" as a79r3.
the parent of the a79r3 is r3.
the X of the a79r3 is "548019.964533955".
the Y of the a79r3 is "1013346.3628308077".

[create a80r3]
a80r3 is a area. "An area (80) in r3".
the printed name of the a80r3 is "Room 3".
Understand "Area 80 in Room 3" as a80r3.
the parent of the a80r3 is r3.
the X of the a80r3 is "547058.8626086956".
the Y of the a80r3 is "1012466.0".

[create a81r3]
a81r3 is a area. "An area (81) in r3".
the printed name of the a81r3 is "Room 3".
Understand "Area 81 in Room 3" as a81r3.
the parent of the a81r3 is r3.
the X of the a81r3 is "547791.7255000001".
the Y of the a81r3 is "1012466.0".

[create a82r3]
a82r3 is a area. "An area (82) in r3".
the printed name of the a82r3 is "Room 3".
Understand "Area 82 in Room 3" as a82r3.
the parent of the a82r3 is r3.
the X of the a82r3 is "547056.8944574267".
the Y of the a82r3 is "1013333.6160476287".

[create d0]
d0 is a door. "A door between a64r1 and a81r3".
d0 is north of a64r1 and south of a81r3.

[create d1]
d1 is a door. "A door between a84r1 and a79r3".
d1 is northwest of a84r1 and southeast of a79r3.

[create d2]
d2 is a door. "A door between a62r1 and a80r3".
d2 is north of a62r1 and south of a80r3.

[create d4]
d4 is a door. "A door between a77r0 and a84r1".
d4 is southwest of a77r0 and northeast of a84r1.

[create d6]
d6 is a door. "A door between a83r1 and a78r2".
d6 is north of a83r1 and south of a78r2.

[create d7]
d7 is a door. "A door between a86r1 and a76r2".
d7 is north of a86r1 and south of a76r2.

southeast of a0r0 is northwest of a1r0.

southwest of a2r0 is northeast of a3r0.

southeast of a4r3 is northwest of a5r3.

southwest of a6r3 is northeast of a7r3.

east of a8r3 is west of a9r3.

west of a10r3 is east of a11r3.

northwest of a12r1 is southeast of a13r1.

southwest of a14r3 is northeast of a15r3.

southeast of a16r3 is northwest of a17r3.

northeast of a18r3 is southwest of a19r3.

northwest of a20r3 is southeast of a21r3.

northeast of a22r0 is southwest of a23r0.

northwest of a24r0 is southeast of a25r0.

southeast of a26r2 is northwest of a27r2.

southwest of a28r1 is northeast of a29r1.

southeast of a30r1 is northwest of a31r1.

southwest of a32r1 is northeast of a33r1.

east of a34r1 is west of a35r1.

west of a36r1 is east of a37r1.

southwest of a38r1 is northeast of a39r1.

south of a40r1 is north of a41r1.

northeast of a42r1 is southwest of a43r1.

northeast of a44r1 is southwest of a45r1.

northwest of a46r1 is southeast of a47r1.

northeast of a48r1 is southwest of a49r1.

east of a50r1 is west of a51r1.

northwest of a52r1 is southeast of a53r1.

southeast of a54r1 is northwest of a55r1.

southwest of a56r2 is northeast of a57r2.

southwest of a33r1 is northeast of a58r1.

north of a35r1 is south of a59r1.

north of a37r1 is south of a60r1.

northeast of a59r1 is southwest of a61r1.

northwest of a60r1 is southeast of a62r1.

west of a39r1 is east of a63r1.

southeast of a41r1 is northwest of a63r1.

northeast of a43r1 is southwest of a64r1.

northwest of a13r1 is southeast of a64r1.

northeast of a45r1 is southwest of a65r1.

northwest of a47r1 is southeast of a66r1.

northeast of a49r1 is southwest of a67r1.

north of a51r1 is south of a68r1.

northwest of a53r1 is southeast of a58r1.

southeast of a55r1 is northwest of a65r1.

southwest of a29r1 is northeast of a66r1.

southeast of a31r1 is northwest of a67r1.

southeast of a1r0 is northwest of a69r0.

southwest of a3r0 is northeast of a70r0.

southeast of a5r3 is northwest of a71r3.

southwest of a7r3 is northeast of a72r3.

south of a9r3 is north of a73r3.

south of a11r3 is north of a74r3.

southwest of a15r3 is northeast of a75r3.

southeast of a17r3 is northwest of a75r3.

northeast of a19r3 is southwest of a71r3.

northwest of a21r3 is southeast of a72r3.

northeast of a23r0 is southwest of a69r0.

northwest of a25r0 is southeast of a70r0.

southeast of a27r2 is northwest of a76r2.

west of a70r0 is east of a77r0.

east of a76r2 is west of a78r2.

southwest of a74r3 is northeast of a79r3.

southwest of a57r2 is northeast of a78r2.

east of a80r3 is west of a81r3.

south of a75r3 is north of a81r3.

north of a80r3 is south of a82r3.

west of a73r3 is east of a82r3.

east of a71r3 is west of a82r3.

west of a72r3 is east of a74r3.

east of a73r3 is west of a79r3.

east of a69r0 is west of a77r0.

north of a61r1 is south of a83r1.

south of a84r1 is north of a85r1.

west of a83r1 is east of a86r1.

east of a65r1 is west of a87r1.

west of a66r1 is east of a87r1.

east of a60r1 is west of a88r1.

east of a61r1 is west of a62r1.

northwest of a68r1 is southeast of a88r1.

south of a63r1 is north of a88r1.

south of a86r1 is north of a87r1.

west of a58r1 is east of a85r1.

west of a59r1 is east of a67r1.

east of a68r1 is west of a85r1.

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
the player is in a83r1.

the orientation of the player is 6.
