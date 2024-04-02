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

[create a2r0]
a2r0 is a area. "An area (2) in r0".
the printed name of the a2r0 is "Room 0".
Understand "Area 2 in Room 0" as a2r0.
the parent of the a2r0 is r0.
the X of the a2r0 is "547672.5949997427".
the Y of the a2r0 is "1575735.8914997524".

[create a3r0]
a3r0 is a area. "An area (3) in r0".
the printed name of the a3r0 is "Room 0".
Understand "Area 3 in Room 0" as a3r0.
the parent of the a3r0 is r0.
the X of the a3r0 is "547812.4375095258".
the Y of the a3r0 is "1575841.0511354979".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "547500.3610336491".
the Y of the a6r0 is "1575752.259346652".

[create a7r0]
a7r0 is a area. "An area (7) in r0".
the printed name of the a7r0 is "Room 0".
Understand "Area 7 in Room 0" as a7r0.
the parent of the a7r0 is r0.
the X of the a7r0 is "547300.0881436878".
the Y of the a7r0 is "1575843.6496814166".

[create a10r0]
a10r0 is a area. "An area (10) in r0".
the printed name of the a10r0 is "Room 0".
Understand "Area 10 in Room 0" as a10r0.
the parent of the a10r0 is r0.
the X of the a10r0 is "548057.915169287".
the Y of the a10r0 is "1575861.610847683".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "548105.0".
the Y of the a22r0 is "1575914.0".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "548073.9123931621".
the Y of the a23r0 is "1575888.166239316".

[create a24r0]
a24r0 is a area. "An area (24) in r0".
the printed name of the a24r0 is "Room 0".
Understand "Area 24 in Room 0" as a24r0.
the parent of the a24r0 is r0.
the X of the a24r0 is "547851.0".
the Y of the a24r0 is "1575914.0".

[create a25r0]
a25r0 is a area. "An area (25) in r0".
the printed name of the a25r0 is "Room 0".
Understand "Area 25 in Room 0" as a25r0.
the parent of the a25r0 is r0.
the X of the a25r0 is "547819.005982906".
the Y of the a25r0 is "1575905.7505982907".

[create a28r0]
a28r0 is a area. "An area (28) in r0".
the printed name of the a28r0 is "Room 0".
Understand "Area 28 in Room 0" as a28r0.
the parent of the a28r0 is r0.
the X of the a28r0 is "547262.1230769231".
the Y of the a28r0 is "1575905.2307692308".

[create a29r0]
a29r0 is a area. "An area (29) in r0".
the printed name of the a29r0 is "Room 0".
Understand "Area 29 in Room 0" as a29r0.
the parent of the a29r0 is r0.
the X of the a29r0 is "547294.8584985461".
the Y of the a29r0 is "1575906.3856886069".

[create a32r0]
a32r0 is a area. "An area (32) in r0".
the printed name of the a32r0 is "Room 0".
Understand "Area 32 in Room 0" as a32r0.
the parent of the a32r0 is r0.
the X of the a32r0 is "547083.0".
the Y of the a32r0 is "1575824.0".

[create a33r0]
a33r0 is a area. "An area (33) in r0".
the printed name of the a33r0 is "Room 0".
Understand "Area 33 in Room 0" as a33r0.
the parent of the a33r0 is r0.
the X of the a33r0 is "547123.7622699386".
the Y of the a33r0 is "1575842.648773006".

[create a36r0]
a36r0 is a area. "An area (36) in r0".
the printed name of the a36r0 is "Room 0".
Understand "Area 36 in Room 0" as a36r0.
the parent of the a36r0 is r0.
the X of the a36r0 is "548114.0".
the Y of the a36r0 is "1575824.0".

[create a37r0]
a37r0 is a area. "An area (37) in r0".
the printed name of the a37r0 is "Room 0".
Understand "Area 37 in Room 0" as a37r0.
the parent of the a37r0 is r0.
the X of the a37r0 is "548074.9448424069".
the Y of the a37r0 is "1575842.8194842406".

[create a42r0]
a42r0 is a area. "An area (42) in r0".
the printed name of the a42r0 is "Room 0".
Understand "Area 42 in Room 0" as a42r0.
the parent of the a42r0 is r0.
the X of the a42r0 is "547092.0".
the Y of the a42r0 is "1575914.0".

[create a43r0]
a43r0 is a area. "An area (43) in r0".
the printed name of the a43r0 is "Room 0".
Understand "Area 43 in Room 0" as a43r0.
the parent of the a43r0 is r0.
the X of the a43r0 is "547119.1180078895".
the Y of the a43r0 is "1575888.5631992111".

[create a70r0]
a70r0 is a area. "An area (70) in r0".
the printed name of the a70r0 is "Room 0".
Understand "Area 70 in Room 0" as a70r0.
the parent of the a70r0 is r0.
the X of the a70r0 is "547138.7741451068".
the Y of the a70r0 is "1575859.9701674308".

[create a13r1]
a13r1 is a area. "An area (13) in r1".
the printed name of the a13r1 is "Room 1".
Understand "Area 13 in Room 1" as a13r1.
the parent of the a13r1 is r1.
the X of the a13r1 is "548244.1730804583".
the Y of the a13r1 is "1574659.7005024815".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "548249.0".
the Y of the a14r1 is "1573968.5778301887".

[create a46r1]
a46r1 is a area. "An area (46) in r1".
the printed name of the a46r1 is "Room 1".
Understand "Area 46 in Room 1" as a46r1.
the parent of the a46r1 is r1.
the X of the a46r1 is "548664.0".
the Y of the a46r1 is "1573791.0".

[create a47r1]
a47r1 is a area. "An area (47) in r1".
the printed name of the a47r1 is "Room 1".
Understand "Area 47 in Room 1" as a47r1.
the parent of the a47r1 is r1.
the X of the a47r1 is "548632.7862838916".
the Y of the a47r1 is "1573816.633971292".

[create a48r1]
a48r1 is a area. "An area (48) in r1".
the printed name of the a48r1 is "Room 1".
Understand "Area 48 in Room 1" as a48r1.
the parent of the a48r1 is r1.
the X of the a48r1 is "548673.0".
the Y of the a48r1 is "1573880.0".

[create a49r1]
a49r1 is a area. "An area (49) in r1".
the printed name of the a49r1 is "Room 1".
Understand "Area 49 in Room 1" as a49r1.
the parent of the a49r1 is r1.
the X of the a49r1 is "548629.9383292026".
the Y of the a49r1 is "1573861.8770228897".

[create a62r1]
a62r1 is a area. "An area (62) in r1".
the printed name of the a62r1 is "Room 1".
Understand "Area 62 in Room 1" as a62r1.
the parent of the a62r1 is r1.
the X of the a62r1 is "547822.0".
the Y of the a62r1 is "1573791.0".

[create a63r1]
a63r1 is a area. "An area (63) in r1".
the printed name of the a63r1 is "Room 1".
Understand "Area 63 in Room 1" as a63r1.
the parent of the a63r1 is r1.
the X of the a63r1 is "547853.2137161084".
the Y of the a63r1 is "1573816.6339712918".

[create a64r1]
a64r1 is a area. "An area (64) in r1".
the printed name of the a64r1 is "Room 1".
Understand "Area 64 in Room 1" as a64r1.
the parent of the a64r1 is r1.
the X of the a64r1 is "547813.0".
the Y of the a64r1 is "1573880.0".

[create a65r1]
a65r1 is a area. "An area (65) in r1".
the printed name of the a65r1 is "Room 1".
Understand "Area 65 in Room 1" as a65r1.
the parent of the a65r1 is r1.
the X of the a65r1 is "547843.8730024241".
the Y of the a65r1 is "1573860.644460919".

[create a72r1]
a72r1 is a area. "An area (72) in r1".
the printed name of the a72r1 is "Room 1".
Understand "Area 72 in Room 1" as a72r1.
the parent of the a72r1 is r1.
the X of the a72r1 is "548615.7290448342".
the Y of the a72r1 is "1573845.0627030537".

[create a76r1]
a76r1 is a area. "An area (76) in r1".
the printed name of the a76r1 is "Room 1".
Understand "Area 76 in Room 1" as a76r1.
the parent of the a76r1 is r1.
the X of the a76r1 is "547867.3945333333".
the Y of the a76r1 is "1573840.2686666665".

[create a8r2]
a8r2 is a area. "An area (8) in r2".
the printed name of the a8r2 is "Room 2".
Understand "Area 8 in Room 2" as a8r2.
the parent of the a8r2 is r2.
the X of the a8r2 is "548535.707266629".
the Y of the a8r2 is "1575861.4777749956".

[create a9r2]
a9r2 is a area. "An area (9) in r2".
the printed name of the a9r2 is "Room 2".
Understand "Area 9 in Room 2" as a9r2.
the parent of the a9r2 is r2.
the X of the a9r2 is "548879.3793965105".
the Y of the a9r2 is "1575774.0644405333".

[create a18r2]
a18r2 is a area. "An area (18) in r2".
the printed name of the a18r2 is "Room 2".
Understand "Area 18 in Room 2" as a18r2.
the parent of the a18r2 is r2.
the X of the a18r2 is "549247.0".
the Y of the a18r2 is "1575914.0".

[create a19r2]
a19r2 is a area. "An area (19) in r2".
the printed name of the a19r2 is "Room 2".
Understand "Area 19 in Room 2" as a19r2.
the parent of the a19r2 is r2.
the X of the a19r2 is "549216.1050372064".
the Y of the a19r2 is "1575887.7894485786".

[create a38r2]
a38r2 is a area. "An area (38) in r2".
the printed name of the a38r2 is "Room 2".
Understand "Area 38 in Room 2" as a38r2.
the parent of the a38r2 is r2.
the X of the a38r2 is "548479.0".
the Y of the a38r2 is "1575824.0".

[create a39r2]
a39r2 is a area. "An area (39) in r2".
the printed name of the a39r2 is "Room 2".
Understand "Area 39 in Room 2" as a39r2.
the parent of the a39r2 is r2.
the X of the a39r2 is "548517.7029286583".
the Y of the a39r2 is "1575842.4774523713".

[create a40r2]
a40r2 is a area. "An area (40) in r2".
the printed name of the a40r2 is "Room 2".
Understand "Area 40 in Room 2" as a40r2.
the parent of the a40r2 is r2.
the X of the a40r2 is "549257.0".
the Y of the a40r2 is "1575824.0".

[create a41r2]
a41r2 is a area. "An area (41) in r2".
the printed name of the a41r2 is "Room 2".
Understand "Area 41 in Room 2" as a41r2.
the parent of the a41r2 is r2.
the X of the a41r2 is "549217.9982014389".
the Y of the a41r2 is "1575842.4442446046".

[create a44r2]
a44r2 is a area. "An area (44) in r2".
the printed name of the a44r2 is "Room 2".
Understand "Area 44 in Room 2" as a44r2.
the parent of the a44r2 is r2.
the X of the a44r2 is "548489.0".
the Y of the a44r2 is "1575914.0".

[create a45r2]
a45r2 is a area. "An area (45) in r2".
the printed name of the a45r2 is "Room 2".
Understand "Area 45 in Room 2" as a45r2.
the parent of the a45r2 is r2.
the X of the a45r2 is "548519.8949627937".
the Y of the a45r2 is "1575887.7894485786".

[create a71r2]
a71r2 is a area. "An area (71) in r2".
the printed name of the a71r2 is "Room 2".
Understand "Area 71 in Room 2" as a71r2.
the parent of the a71r2 is r2.
the X of the a71r2 is "549200.3051002843".
the Y of the a71r2 is "1575861.4983535397".

[create a11r3]
a11r3 is a area. "An area (11) in r3".
the printed name of the a11r3 is "Room 3".
Understand "Area 11 in Room 3" as a11r3.
the parent of the a11r3 is r3.
the X of the a11r3 is "546411.7555675791".
the Y of the a11r3 is "1574499.993592572".

[create a12r3]
a12r3 is a area. "An area (12) in r3".
the printed name of the a12r3 is "Room 3".
Understand "Area 12 in Room 3" as a12r3.
the parent of the a12r3 is r3.
the X of the a12r3 is "546745.1956102175".
the Y of the a12r3 is "1573933.3441753672".

[create a15r3]
a15r3 is a area. "An area (15) in r3".
the printed name of the a15r3 is "Room 3".
Understand "Area 15 in Room 3" as a15r3.
the parent of the a15r3 is r3.
the X of the a15r3 is "546314.4927937491".
the Y of the a15r3 is "1574179.842658367".

[create a16r3]
a16r3 is a area. "An area (16) in r3".
the printed name of the a16r3 is "Room 3".
Understand "Area 16 in Room 3" as a16r3.
the parent of the a16r3 is r3.
the X of the a16r3 is "546404.5615324359".
the Y of the a16r3 is "1574497.5318836719".

[create a50r3]
a50r3 is a area. "An area (50) in r3".
the printed name of the a50r3 is "Room 3".
Understand "Area 50 in Room 3" as a50r3.
the parent of the a50r3 is r3.
the X of the a50r3 is "546109.0".
the Y of the a50r3 is "1573769.0".

[create a51r3]
a51r3 is a area. "An area (51) in r3".
the printed name of the a51r3 is "Room 3".
Understand "Area 51 in Room 3" as a51r3.
the parent of the a51r3 is r3.
the X of the a51r3 is "546070.1111111111".
the Y of the a51r3 is "1573863.1566122049".

[create a52r3]
a52r3 is a area. "An area (52) in r3".
the printed name of the a52r3 is "Room 3".
Understand "Area 52 in Room 3" as a52r3.
the parent of the a52r3 is r3.
the X of the a52r3 is "546941.0".
the Y of the a52r3 is "1573800.0".

[create a53r3]
a53r3 is a area. "An area (53) in r3".
the printed name of the a53r3 is "Room 3".
Understand "Area 53 in Room 3" as a53r3.
the parent of the a53r3 is r3.
the X of the a53r3 is "546898.4517163367".
the Y of the a53r3 is "1573849.4252993807".

[create a54r3]
a54r3 is a area. "An area (54) in r3".
the printed name of the a54r3 is "Room 3".
Understand "Area 54 in Room 3" as a54r3.
the parent of the a54r3 is r3.
the X of the a54r3 is "546475.0".
the Y of the a54r3 is "1574110.0".

[create a55r3]
a55r3 is a area. "An area (55) in r3".
the printed name of the a55r3 is "Room 3".
Understand "Area 55 in Room 3" as a55r3.
the parent of the a55r3 is r3.
the X of the a55r3 is "546392.5780853999".
the Y of the a55r3 is "1574146.0479452054".

[create a56r3]
a56r3 is a area. "An area (56) in r3".
the printed name of the a56r3 is "Room 3".
Understand "Area 56 in Room 3" as a56r3.
the parent of the a56r3 is r3.
the X of the a56r3 is "545197.0".
the Y of the a56r3 is "1574331.0".

[create a57r3]
a57r3 is a area. "An area (57) in r3".
the printed name of the a57r3 is "Room 3".
Understand "Area 57 in Room 3" as a57r3.
the parent of the a57r3 is r3.
the X of the a57r3 is "545236.6586345376".
the Y of the a57r3 is "1574297.8333333335".

[create a58r3]
a58r3 is a area. "An area (58) in r3".
the printed name of the a58r3 is "Room 3".
Understand "Area 58 in Room 3" as a58r3.
the parent of the a58r3 is r3.
the X of the a58r3 is "545280.0".
the Y of the a58r3 is "1574368.0".

[create a59r3]
a59r3 is a area. "An area (59) in r3".
the printed name of the a59r3 is "Room 3".
Understand "Area 59 in Room 3" as a59r3.
the parent of the a59r3 is r3.
the X of the a59r3 is "545270.4164161286".
the Y of the a59r3 is "1574333.7010124677".

[create a66r3]
a66r3 is a area. "An area (66) in r3".
the printed name of the a66r3 is "Room 3".
Understand "Area 66 in Room 3" as a66r3.
the parent of the a66r3 is r3.
the X of the a66r3 is "546640.0".
the Y of the a66r3 is "1573772.0".

[create a67r3]
a67r3 is a area. "An area (67) in r3".
the printed name of the a67r3 is "Room 3".
Understand "Area 67 in Room 3" as a67r3.
the parent of the a67r3 is r3.
the X of the a67r3 is "546673.3111530826".
the Y of the a67r3 is "1573821.8738543612".

[create a68r3]
a68r3 is a area. "An area (68) in r3".
the printed name of the a68r3 is "Room 3".
Understand "Area 68 in Room 3" as a68r3.
the parent of the a68r3 is r3.
the X of the a68r3 is "545147.0".
the Y of the a68r3 is "1573880.0".

[create a69r3]
a69r3 is a area. "An area (69) in r3".
the printed name of the a69r3 is "Room 3".
Understand "Area 69 in Room 3" as a69r3.
the parent of the a69r3 is r3.
the X of the a69r3 is "545195.4282420991".
the Y of the a69r3 is "1573903.1313011711".

[create a73r3]
a73r3 is a area. "An area (73) in r3".
the printed name of the a73r3 is "Room 3".
Understand "Area 73 in Room 3" as a73r3.
the parent of the a73r3 is r3.
the X of the a73r3 is "546062.7882951223".
the Y of the a73r3 is "1573882.7079821012".

[create a74r3]
a74r3 is a area. "An area (74) in r3".
the printed name of the a74r3 is "Room 3".
Understand "Area 74 in Room 3" as a74r3.
the parent of the a74r3 is r3.
the X of the a74r3 is "546834.9629832632".
the Y of the a74r3 is "1573928.2779362001".

[create a75r3]
a75r3 is a area. "An area (75) in r3".
the printed name of the a75r3 is "Room 3".
Understand "Area 75 in Room 3" as a75r3.
the parent of the a75r3 is r3.
the X of the a75r3 is "545239.2405909919".
the Y of the a75r3 is "1574296.5702544395".

[create a77r3]
a77r3 is a area. "An area (77) in r3".
the printed name of the a77r3 is "Room 3".
Understand "Area 77 in Room 3" as a77r3.
the parent of the a77r3 is r3.
the X of the a77r3 is "545297.9453007281".
the Y of the a77r3 is "1573957.8516632349".

[create a0r4]
a0r4 is a area. "An area (0) in r4".
the printed name of the a0r4 is "Room 4".
Understand "Area 0 in Room 4" as a0r4.
the parent of the a0r4 is r4.
the X of the a0r4 is "545322.2765708307".
the Y of the a0r4 is "1575541.3817745447".

[create a1r4]
a1r4 is a area. "An area (1) in r4".
the printed name of the a1r4 is "Room 4".
Understand "Area 1 in Room 4" as a1r4.
the parent of the a1r4 is r4.
the X of the a1r4 is "545568.7588065893".
the Y of the a1r4 is "1575838.1469224335".

[create a4r4]
a4r4 is a area. "An area (4) in r4".
the printed name of the a4r4 is "Room 4".
Understand "Area 4 in Room 4" as a4r4.
the parent of the a4r4 is r4.
the X of the a4r4 is "546152.4076643826".
the Y of the a4r4 is "1575732.5462389272".

[create a5r4]
a5r4 is a area. "An area (5) in r4".
the printed name of the a5r4 is "Room 4".
Understand "Area 5 in Room 4" as a5r4.
the parent of the a5r4 is r4.
the X of the a5r4 is "546497.6927655563".
the Y of the a5r4 is "1575861.1041153662".

[create a17r4]
a17r4 is a area. "An area (17) in r4".
the printed name of the a17r4 is "Room 4".
Understand "Area 17 in Room 4" as a17r4.
the parent of the a17r4 is r4.
the X of the a17r4 is "545343.3590770198".
the Y of the a17r4 is "1575177.3054104475".

[create a20r4]
a20r4 is a area. "An area (20) in r4".
the printed name of the a20r4 is "Room 4".
Understand "Area 20 in Room 4" as a20r4.
the parent of the a20r4 is r4.
the X of the a20r4 is "546545.0".
the Y of the a20r4 is "1575914.0".

[create a21r4]
a21r4 is a area. "An area (21) in r4".
the printed name of the a21r4 is "Room 4".
Understand "Area 21 in Room 4" as a21r4.
the parent of the a21r4 is r4.
the X of the a21r4 is "546514.0846219931".
the Y of the a21r4 is "1575888.1834621993".

[create a26r4]
a26r4 is a area. "An area (26) in r4".
the printed name of the a26r4 is "Room 4".
Understand "Area 26 in Room 4" as a26r4.
the parent of the a26r4 is r4.
the X of the a26r4 is "545531.0".
the Y of the a26r4 is "1575932.0".

[create a27r4]
a27r4 is a area. "An area (27) in r4".
the printed name of the a27r4 is "Room 4".
Understand "Area 27 in Room 4" as a27r4.
the parent of the a27r4 is r4.
the X of the a27r4 is "545566.3749999998".
the Y of the a27r4 is "1575844.2712765958".

[create a30r4]
a30r4 is a area. "An area (30) in r4".
the printed name of the a30r4 is "Room 4".
Understand "Area 30 in Room 4" as a30r4.
the parent of the a30r4 is r4.
the X of the a30r4 is "545181.0".
the Y of the a30r4 is "1575603.0".

[create a31r4]
a31r4 is a area. "An area (31) in r4".
the printed name of the a31r4 is "Room 4".
Understand "Area 31 in Room 4" as a31r4.
the parent of the a31r4 is r4.
the X of the a31r4 is "545263.1833333333".
the Y of the a31r4 is "1575565.6666666667".

[create a34r4]
a34r4 is a area. "An area (34) in r4".
the printed name of the a34r4 is "Room 4".
Understand "Area 34 in Room 4" as a34r4.
the parent of the a34r4 is r4.
the X of the a34r4 is "546554.0".
the Y of the a34r4 is "1575824.0".

[create a35r4]
a35r4 is a area. "An area (35) in r4".
the printed name of the a35r4 is "Room 4".
Understand "Area 35 in Room 4" as a35r4.
the parent of the a35r4 is r4.
the X of the a35r4 is "546513.5982564336".
the Y of the a35r4 is "1575842.6848256432".

[create a60r4]
a60r4 is a area. "An area (60) in r4".
the printed name of the a60r4 is "Room 4".
Understand "Area 60 in Room 4" as a60r4.
the parent of the a60r4 is r4.
the X of the a60r4 is "545181.0".
the Y of the a60r4 is "1575090.0".

[create a61r4]
a61r4 is a area. "An area (61) in r4".
the printed name of the a61r4 is "Room 4".
Understand "Area 61 in Room 4" as a61r4.
the parent of the a61r4 is r4.
the X of the a61r4 is "545244.7508833924".
the Y of the a61r4 is "1575126.125".

[create d0]
d0 is a door. "A door between a6r0 and a74r3".
d0 is south of a6r0 and north of a74r3.

[create d2]
d2 is a door. "A door between a16r3 and a17r4".
d2 is northwest of a16r3 and southeast of a17r4.

[create d3]
d3 is a door. "A door between a11r3 and a4r4".
d3 is north of a11r3 and south of a4r4.

[create d6]
d6 is a door. "A door between a13r1 and a9r2".
d6 is northeast of a13r1 and southwest of a9r2.

[create d8]
d8 is a door. "A door between a2r0 and a13r1".
d8 is southeast of a2r0 and northwest of a13r1.

northeast of a0r4 is southwest of a1r4.

northeast of a2r0 is southwest of a3r0.

east of a4r4 is west of a5r4.

west of a2r0 is east of a6r0.

east of a1r4 is west of a4r4.

northwest of a6r0 is southeast of a7r0.

east of a8r2 is west of a9r2.

east of a3r0 is west of a10r0.

southeast of a11r3 is northwest of a12r3.

south of a13r1 is north of a14r1.

north of a15r3 is south of a16r3.

south of a0r4 is north of a17r4.

southwest of a18r2 is northeast of a19r2.

southwest of a20r4 is northeast of a21r4.

southwest of a22r0 is northeast of a23r0.

west of a24r0 is east of a25r0.

south of a26r4 is north of a27r4.

east of a28r0 is west of a29r0.

southeast of a30r4 is northwest of a31r4.

northeast of a32r0 is southwest of a33r0.

northwest of a34r4 is southeast of a35r4.

northwest of a36r0 is southeast of a37r0.

northeast of a38r2 is southwest of a39r2.

northwest of a40r2 is southeast of a41r2.

southeast of a42r0 is northwest of a43r0.

southeast of a44r2 is northwest of a45r2.

northwest of a46r1 is southeast of a47r1.

southwest of a48r1 is northeast of a49r1.

north of a50r3 is south of a51r3.

northwest of a52r3 is southeast of a53r3.

northwest of a54r3 is southeast of a55r3.

southeast of a56r3 is northwest of a57r3.

south of a58r3 is north of a59r3.

northeast of a60r4 is southwest of a61r4.

northeast of a62r1 is southwest of a63r1.

southeast of a64r1 is northwest of a65r1.

northeast of a66r3 is southwest of a67r3.

northeast of a68r3 is southwest of a69r3.

west of a0r4 is east of a31r4.

northeast of a33r0 is southwest of a70r0.

southeast of a5r4 is northwest of a35r4.

southeast of a10r0 is northwest of a37r0.

southwest of a8r2 is northeast of a39r2.

northwest of a41r2 is southeast of a71r2.

southeast of a43r0 is northwest of a70r0.

northwest of a8r2 is southeast of a45r2.

southwest of a19r2 is northeast of a71r2.

southwest of a49r1 is northeast of a72r1.

north of a51r3 is south of a73r3.

northwest of a53r3 is southeast of a74r3.

southeast of a15r3 is northwest of a55r3.

southeast of a57r3 is northwest of a75r3.

southwest of a59r3 is northeast of a75r3.

west of a11r3 is east of a16r3.

southwest of a17r4 is northeast of a61r4.

northeast of a63r1 is southwest of a76r1.

southeast of a65r1 is northwest of a76r1.

northwest of a47r1 is southeast of a72r1.

northeast of a5r4 is southwest of a21r4.

northeast of a10r0 is southwest of a23r0.

north of a3r0 is south of a25r0.

north of a1r4 is south of a27r4.

north of a7r0 is south of a29r0.

west of a14r1 is east of a76r1.

west of a73r3 is east of a77r3.

southwest of a15r3 is northeast of a73r3.

southwest of a12r3 is northeast of a67r3.

east of a12r3 is west of a74r3.

northeast of a69r3 is southwest of a77r3.

south of a75r3 is north of a77r3.

east of a14r1 is west of a72r1.

east of a9r2 is west of a71r2.

west of a7r0 is east of a70r0.

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
the player is in a0r4.

the orientation of the player is 6.
