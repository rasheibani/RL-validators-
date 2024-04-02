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

[create a40r0]
a40r0 is a area. "An area (40) in r0".
the printed name of the a40r0 is "Room 0".
Understand "Area 40 in Room 0" as a40r0.
the parent of the a40r0 is r0.
the X of the a40r0 is "545456.3319092895".
the Y of the a40r0 is "1399129.437845304".

[create a41r0]
a41r0 is a area. "An area (41) in r0".
the printed name of the a41r0 is "Room 0".
Understand "Area 41 in Room 0" as a41r0.
the parent of the a41r0 is r0.
the X of the a41r0 is "545811.8228866081".
the Y of the a41r0 is "1399040.0615040062".

[create a42r0]
a42r0 is a area. "An area (42) in r0".
the printed name of the a42r0 is "Room 0".
Understand "Area 42 in Room 0" as a42r0.
the parent of the a42r0 is r0.
the X of the a42r0 is "545127.3913043479".
the Y of the a42r0 is "1399045.5".

[create a69r0]
a69r0 is a area. "An area (69) in r0".
the printed name of the a69r0 is "Room 0".
Understand "Area 69 in Room 0" as a69r0.
the parent of the a69r0 is r0.
the X of the a69r0 is "545857.0".
the Y of the a69r0 is "1398991.0".

[create a70r0]
a70r0 is a area. "An area (70) in r0".
the printed name of the a70r0 is "Room 0".
Understand "Area 70 in Room 0" as a70r0.
the parent of the a70r0 is r0.
the X of the a70r0 is "545825.9579422104".
the Y of the a70r0 is "1399016.6166125853".

[create a73r0]
a73r0 is a area. "An area (73) in r0".
the printed name of the a73r0 is "Room 0".
Understand "Area 73 in Room 0" as a73r0.
the parent of the a73r0 is r0.
the X of the a73r0 is "545866.0".
the Y of the a73r0 is "1399080.0".

[create a74r0]
a74r0 is a area. "An area (74) in r0".
the printed name of the a74r0 is "Room 0".
Understand "Area 74 in Room 0" as a74r0.
the parent of the a74r0 is r0.
the X of the a74r0 is "545838.8491587283".
the Y of the a74r0 is "1399060.2680626006".

[create a77r0]
a77r0 is a area. "An area (77) in r0".
the printed name of the a77r0 is "Room 0".
Understand "Area 77 in Room 0" as a77r0.
the parent of the a77r0 is r0.
the X of the a77r0 is "545080.0".
the Y of the a77r0 is "1398991.0".

[create a78r0]
a78r0 is a area. "An area (78) in r0".
the printed name of the a78r0 is "Room 0".
Understand "Area 78 in Room 0" as a78r0.
the parent of the a78r0 is r0.
the X of the a78r0 is "545110.6227296316".
the Y of the a78r0 is "1399016.5742086142".

[create a79r0]
a79r0 is a area. "An area (79) in r0".
the printed name of the a79r0 is "Room 0".
Understand "Area 79 in Room 0" as a79r0.
the parent of the a79r0 is r0.
the X of the a79r0 is "545071.0".
the Y of the a79r0 is "1399080.0".

[create a80r0]
a80r0 is a area. "An area (80) in r0".
the printed name of the a80r0 is "Room 0".
Understand "Area 80 in Room 0" as a80r0.
the parent of the a80r0 is r0.
the X of the a80r0 is "545113.2240682533".
the Y of the a80r0 is "1399061.792321509".

[create a5r1]
a5r1 is a area. "An area (5) in r1".
the printed name of the a5r1 is "Room 1".
Understand "Area 5 in Room 1" as a5r1.
the parent of the a5r1 is r1.
the X of the a5r1 is "545559.0257754212".
the Y of the a5r1 is "1400926.639949738".

[create a6r1]
a6r1 is a area. "An area (6) in r1".
the printed name of the a6r1 is "Room 1".
Understand "Area 6 in Room 1" as a6r1.
the parent of the a6r1 is r1.
the X of the a6r1 is "545582.0735241217".
the Y of the a6r1 is "1400811.7267338363".

[create a7r1]
a7r1 is a area. "An area (7) in r1".
the printed name of the a7r1 is "Room 1".
Understand "Area 7 in Room 1" as a7r1.
the parent of the a7r1 is r1.
the X of the a7r1 is "545189.1884979303".
the Y of the a7r1 is "1401059.7787601026".

[create a8r1]
a8r1 is a area. "An area (8) in r1".
the printed name of the a8r1 is "Room 1".
Understand "Area 8 in Room 1" as a8r1.
the parent of the a8r1 is r1.
the X of the a8r1 is "545641.8999999999".
the Y of the a8r1 is "1401048.1052980134".

[create a43r1]
a43r1 is a area. "An area (43) in r1".
the printed name of the a43r1 is "Room 1".
Understand "Area 43 in Room 1" as a43r1.
the parent of the a43r1 is r1.
the X of the a43r1 is "545132.0".
the Y of the a43r1 is "1401024.0".

[create a44r1]
a44r1 is a area. "An area (44) in r1".
the printed name of the a44r1 is "Room 1".
Understand "Area 44 in Room 1" as a44r1.
the parent of the a44r1 is r1.
the X of the a44r1 is "545174.2698675497".
the Y of the a44r1 is "1401042.4980132452".

[create a47r1]
a47r1 is a area. "An area (47) in r1".
the printed name of the a47r1 is "Room 1".
Understand "Area 47 in Room 1" as a47r1.
the parent of the a47r1 is r1.
the X of the a47r1 is "545141.0".
the Y of the a47r1 is "1401114.0".

[create a48r1]
a48r1 is a area. "An area (48) in r1".
the printed name of the a48r1 is "Room 1".
Understand "Area 48 in Room 1" as a48r1.
the parent of the a48r1 is r1.
the X of the a48r1 is "545172.0876068376".
the Y of the a48r1 is "1401088.166239316".

[create a84r1]
a84r1 is a area. "An area (84) in r1".
the printed name of the a84r1 is "Room 1".
Understand "Area 84 in Room 1" as a84r1.
the parent of the a84r1 is r1.
the X of the a84r1 is "545676.0".
the Y of the a84r1 is "1401104.0".

[create a27r2]
a27r2 is a area. "An area (27) in r2".
the printed name of the a27r2 is "Room 2".
Understand "Area 27 in Room 2" as a27r2.
the parent of the a27r2 is r2.
the X of the a27r2 is "546357.7606867261".
the Y of the a27r2 is "1399149.5866818968".

[create a38r2]
a38r2 is a area. "An area (38) in r2".
the printed name of the a38r2 is "Room 2".
Understand "Area 38 in Room 2" as a38r2.
the parent of the a38r2 is r2.
the X of the a38r2 is "546292.5661679945".
the Y of the a38r2 is "1399019.9341592467".

[create a39r2]
a39r2 is a area. "An area (39) in r2".
the printed name of the a39r2 is "Room 2".
Understand "Area 39 in Room 2" as a39r2.
the parent of the a39r2 is r2.
the X of the a39r2 is "546355.0091290362".
the Y of the a39r2 is "1399130.1696830264".

[create a57r2]
a57r2 is a area. "An area (57) in r2".
the printed name of the a57r2 is "Room 2".
Understand "Area 57 in Room 2" as a57r2.
the parent of the a57r2 is r2.
the X of the a57r2 is "546459.0".
the Y of the a57r2 is "1399003.0".

[create a58r2]
a58r2 is a area. "An area (58) in r2".
the printed name of the a58r2 is "Room 2".
Understand "Area 58 in Room 2" as a58r2.
the parent of the a58r2 is r2.
the X of the a58r2 is "546415.8503889912".
the Y of the a58r2 is "1399061.502728187".

[create a83r2]
a83r2 is a area. "An area (83) in r2".
the printed name of the a83r2 is "Room 2".
Understand "Area 83 in Room 2" as a83r2.
the parent of the a83r2 is r2.
the X of the a83r2 is "546269.0".
the Y of the a83r2 is "1398969.0".

[create a0r3]
a0r3 is a area. "An area (0) in r3".
the printed name of the a0r3 is "Room 3".
Understand "Area 0 in Room 3" as a0r3.
the parent of the a0r3 is r3.
the X of the a0r3 is "548581.0854657069".
the Y of the a0r3 is "1400759.569250254".

[create a1r3]
a1r3 is a area. "An area (1) in r3".
the printed name of the a1r3 is "Room 3".
Understand "Area 1 in Room 3" as a1r3.
the parent of the a1r3 is r3.
the X of the a1r3 is "548581.25583807".
the Y of the a1r3 is "1400513.8317766765".

[create a2r3]
a2r3 is a area. "An area (2) in r3".
the printed name of the a2r3 is "Room 3".
Understand "Area 2 in Room 3" as a2r3.
the parent of the a2r3 is r3.
the X of the a2r3 is "547634.8727524201".
the Y of the a2r3 is "1400956.6882780083".

[create a3r3]
a3r3 is a area. "An area (3) in r3".
the printed name of the a3r3 is "Room 3".
Understand "Area 3 in Room 3" as a3r3.
the parent of the a3r3 is r3.
the X of the a3r3 is "547228.9001965043".
the Y of the a3r3 is "1400793.9469272909".

[create a4r3]
a4r3 is a area. "An area (4) in r3".
the printed name of the a4r3 is "Room 3".
Understand "Area 4 in Room 3" as a4r3.
the parent of the a4r3 is r3.
the X of the a4r3 is "547259.0289071669".
the Y of the a4r3 is "1400929.5602931618".

[create a9r3]
a9r3 is a area. "An area (9) in r3".
the printed name of the a9r3 is "Room 3".
Understand "Area 9 in Room 3" as a9r3.
the parent of the a9r3 is r3.
the X of the a9r3 is "547275.0150019894".
the Y of the a9r3 is "1400941.2675300585".

[create a10r3]
a10r3 is a area. "An area (10) in r3".
the printed name of the a10r3 is "Room 3".
Understand "Area 10 in Room 3" as a10r3.
the parent of the a10r3 is r3.
the X of the a10r3 is "547614.7307586102".
the Y of the a10r3 is "1400940.3229080373".

[create a16r3]
a16r3 is a area. "An area (16) in r3".
the printed name of the a16r3 is "Room 3".
Understand "Area 16 in Room 3" as a16r3.
the parent of the a16r3 is r3.
the X of the a16r3 is "547610.7426007325".
the Y of the a16r3 is "1400061.75".

[create a17r3]
a17r3 is a area. "An area (17) in r3".
the printed name of the a17r3 is "Room 3".
Understand "Area 17 in Room 3" as a17r3.
the parent of the a17r3 is r3.
the X of the a17r3 is "547615.3784065933".
the Y of the a17r3 is "1400079.5".

[create a18r3]
a18r3 is a area. "An area (18) in r3".
the printed name of the a18r3 is "Room 3".
Understand "Area 18 in Room 3" as a18r3.
the parent of the a18r3 is r3.
the X of the a18r3 is "548312.6801725186".
the Y of the a18r3 is "1400127.2069072586".

[create a45r3]
a45r3 is a area. "An area (45) in r3".
the printed name of the a45r3 is "Room 3".
Understand "Area 45 in Room 3" as a45r3.
the parent of the a45r3 is r3.
the X of the a45r3 is "547138.0".
the Y of the a45r3 is "1401104.0".

[create a46r3]
a46r3 is a area. "An area (46) in r3".
the printed name of the a46r3 is "Room 3".
Understand "Area 46 in Room 3" as a46r3.
the parent of the a46r3 is r3.
the X of the a46r3 is "547172.5".
the Y of the a46r3 is "1401047.94392936".

[create a49r3]
a49r3 is a area. "An area (49) in r3".
the printed name of the a49r3 is "Room 3".
Understand "Area 49 in Room 3" as a49r3.
the parent of the a49r3 is r3.
the X of the a49r3 is "547207.0".
the Y of the a49r3 is "1401104.0".

[create a50r3]
a50r3 is a area. "An area (50) in r3".
the printed name of the a50r3 is "Room 3".
Understand "Area 50 in Room 3" as a50r3.
the parent of the a50r3 is r3.
the X of the a50r3 is "547239.8264604809".
the Y of the a50r3 is "1401105.7673539517".

[create a51r3]
a51r3 is a area. "An area (51) in r3".
the printed name of the a51r3 is "Room 3".
Understand "Area 51 in Room 3" as a51r3.
the parent of the a51r3 is r3.
the X of the a51r3 is "547683.0".
the Y of the a51r3 is "1401104.0".

[create a52r3]
a52r3 is a area. "An area (52) in r3".
the printed name of the a52r3 is "Room 3".
Understand "Area 52 in Room 3" as a52r3.
the parent of the a52r3 is r3.
the X of the a52r3 is "547650.0059829061".
the Y of the a52r3 is "1401105.7505982905".

[create a63r3]
a63r3 is a area. "An area (63) in r3".
the printed name of the a63r3 is "Room 3".
Understand "Area 63 in Room 3" as a63r3.
the parent of the a63r3 is r3.
the X of the a63r3 is "548725.0".
the Y of the a63r3 is "1400471.0".

[create a64r3]
a64r3 is a area. "An area (64) in r3".
the printed name of the a64r3 is "Room 3".
Understand "Area 64 in Room 3" as a64r3.
the parent of the a64r3 is r3.
the X of the a64r3 is "548604.1889534884".
the Y of the a64r3 is "1400506.5".

[create a65r3]
a65r3 is a area. "An area (65) in r3".
the printed name of the a65r3 is "Room 3".
Understand "Area 65 in Room 3" as a65r3.
the parent of the a65r3 is r3.
the X of the a65r3 is "548725.0".
the Y of the a65r3 is "1400822.0".

[create a66r3]
a66r3 is a area. "An area (66) in r3".
the printed name of the a66r3 is "Room 3".
Understand "Area 66 in Room 3" as a66r3.
the parent of the a66r3 is r3.
the X of the a66r3 is "548645.265503876".
the Y of the a66r3 is "1400787.1666666665".

[create a67r3]
a67r3 is a area. "An area (67) in r3".
the printed name of the a67r3 is "Room 3".
Understand "Area 67 in Room 3" as a67r3.
the parent of the a67r3 is r3.
the X of the a67r3 is "547444.4105263158".
the Y of the a67r3 is "1400944.1684210526".

[create a68r3]
a68r3 is a area. "An area (68) in r3".
the printed name of the a68r3 is "Room 3".
Understand "Area 68 in Room 3" as a68r3.
the parent of the a68r3 is r3.
the X of the a68r3 is "547434.5323545705".
the Y of the a68r3 is "1400916.726398892".

[create a85r3]
a85r3 is a area. "An area (85) in r3".
the printed name of the a85r3 is "Room 3".
Understand "Area 85 in Room 3" as a85r3.
the parent of the a85r3 is r3.
the X of the a85r3 is "547257.4801589376".
the Y of the a85r3 is "1400932.761109078".

[create a11r4]
a11r4 is a area. "An area (11) in r4".
the printed name of the a11r4 is "Room 4".
Understand "Area 11 in Room 4" as a11r4.
the parent of the a11r4 is r4.
the X of the a11r4 is "547313.3140968417".
the Y of the a11r4 is "1399929.9661116619".

[create a12r4]
a12r4 is a area. "An area (12) in r4".
the printed name of the a12r4 is "Room 4".
Understand "Area 12 in Room 4" as a12r4.
the parent of the a12r4 is r4.
the X of the a12r4 is "547315.3587671968".
the Y of the a12r4 is "1399853.7479421725".

[create a13r4]
a13r4 is a area. "An area (13) in r4".
the printed name of the a13r4 is "Room 4".
Understand "Area 13 in Room 4" as a13r4.
the parent of the a13r4 is r4.
the X of the a13r4 is "547610.4046474359".
the Y of the a13r4 is "1399907.5500000003".

[create a14r4]
a14r4 is a area. "An area (14) in r4".
the printed name of the a14r4 is "Room 4".
Understand "Area 14 in Room 4" as a14r4.
the parent of the a14r4 is r4.
the X of the a14r4 is "547610.5696394232".
the Y of the a14r4 is "1399984.65".

[create a15r4]
a15r4 is a area. "An area (15) in r4".
the printed name of the a15r4 is "Room 4".
Understand "Area 15 in Room 4" as a15r4.
the parent of the a15r4 is r4.
the X of the a15r4 is "547311.2547552986".
the Y of the a15r4 is "1400006.1835343419".

[create a19r4]
a19r4 is a area. "An area (19) in r4".
the printed name of the a19r4 is "Room 4".
Understand "Area 19 in Room 4" as a19r4.
the parent of the a19r4 is r4.
the X of the a19r4 is "547611.0264498447".
the Y of the a19r4 is "1399608.7685185184".

[create a20r4]
a20r4 is a area. "An area (20) in r4".
the printed name of the a20r4 is "Room 4".
Understand "Area 20 in Room 4" as a20r4.
the parent of the a20r4 is r4.
the X of the a20r4 is "547610.8731033735".
the Y of the a20r4 is "1399683.1203703703".

[create a21r4]
a21r4 is a area. "An area (21) in r4".
the printed name of the a21r4 is "Room 4".
Understand "Area 21 in Room 4" as a21r4.
the parent of the a21r4 is r4.
the X of the a21r4 is "547319.4064301433".
the Y of the a21r4 is "1399701.3094816657".

[create a22r4]
a22r4 is a area. "An area (22) in r4".
the printed name of the a22r4 is "Room 4".
Understand "Area 22 in Room 4" as a22r4.
the parent of the a22r4 is r4.
the X of the a22r4 is "547321.410514174".
the Y of the a22r4 is "1399625.089246206".

[create a23r4]
a23r4 is a area. "An area (23) in r4".
the printed name of the a23r4 is "Room 4".
Understand "Area 23 in Room 4" as a23r4.
the parent of the a23r4 is r4.
the X of the a23r4 is "547610.7256942875".
the Y of the a23r4 is "1399757.4722222222".

[create a24r4]
a24r4 is a area. "An area (24) in r4".
the printed name of the a24r4 is "Room 4".
Understand "Area 24 in Room 4" as a24r4.
the parent of the a24r4 is r4.
the X of the a24r4 is "547317.3893581654".
the Y of the a24r4 is "1399777.5290559984".

[create a25r4]
a25r4 is a area. "An area (25) in r4".
the printed name of the a25r4 is "Room 4".
Understand "Area 25 in Room 4" as a25r4.
the parent of the a25r4 is r4.
the X of the a25r4 is "547610.5842225868".
the Y of the a25r4 is "1399831.824074074".

[create a26r4]
a26r4 is a area. "An area (26) in r4".
the printed name of the a26r4 is "Room 4".
Understand "Area 26 in Room 4" as a26r4.
the parent of the a26r4 is r4.
the X of the a26r4 is "548715.0956758843".
the Y of the a26r4 is "1399764.6963070636".

[create a28r4]
a28r4 is a area. "An area (28) in r4".
the printed name of the a28r4 is "Room 4".
Understand "Area 28 in Room 4" as a28r4.
the parent of the a28r4 is r4.
the X of the a28r4 is "547391.6749248416".
the Y of the a28r4 is "1399219.4923790898".

[create a29r4]
a29r4 is a area. "An area (29) in r4".
the printed name of the a29r4 is "Room 4".
Understand "Area 29 in Room 4" as a29r4.
the parent of the a29r4 is r4.
the X of the a29r4 is "547558.469293531".
the Y of the a29r4 is "1399219.674768928".

[create a30r4]
a30r4 is a area. "An area (30) in r4".
the printed name of the a30r4 is "Room 4".
Understand "Area 30 in Room 4" as a30r4.
the parent of the a30r4 is r4.
the X of the a30r4 is "547325.3817070527".
the Y of the a30r4 is "1399472.6468931339".

[create a31r4]
a31r4 is a area. "An area (31) in r4".
the printed name of the a31r4 is "Room 4".
Understand "Area 31 in Room 4" as a31r4.
the parent of the a31r4 is r4.
the X of the a31r4 is "547611.1857337016".
the Y of the a31r4 is "1399534.4166666667".

[create a32r4]
a32r4 is a area. "An area (32) in r4".
the printed name of the a32r4 is "Room 4".
Understand "Area 32 in Room 4" as a32r4.
the parent of the a32r4 is r4.
the X of the a32r4 is "548714.3458504477".
the Y of the a32r4 is "1399420.6181536037".

[create a33r4]
a33r4 is a area. "An area (33) in r4".
the printed name of the a33r4 is "Room 4".
Understand "Area 33 in Room 4" as a33r4.
the parent of the a33r4 is r4.
the X of the a33r4 is "547323.4021138479".
the Y of the a33r4 is "1399548.8683752534".

[create a34r4]
a34r4 is a area. "An area (34) in r4".
the printed name of the a34r4 is "Room 4".
Understand "Area 34 in Room 4" as a34r4.
the parent of the a34r4 is r4.
the X of the a34r4 is "546952.9132133439".
the Y of the a34r4 is "1399040.22214224".

[create a35r4]
a35r4 is a area. "An area (35) in r4".
the printed name of the a35r4 is "Room 4".
Understand "Area 35 in Room 4" as a35r4.
the parent of the a35r4 is r4.
the X of the a35r4 is "547225.8263506808".
the Y of the a35r4 is "1399083.8681768954".

[create a36r4]
a36r4 is a area. "An area (36) in r4".
the printed name of the a36r4 is "Room 4".
Understand "Area 36 in Room 4" as a36r4.
the parent of the a36r4 is r4.
the X of the a36r4 is "547708.414830073".
the Y of the a36r4 is "1399094.190476427".

[create a37r4]
a37r4 is a area. "An area (37) in r4".
the printed name of the a37r4 is "Room 4".
Understand "Area 37 in Room 4" as a37r4.
the parent of the a37r4 is r4.
the X of the a37r4 is "548433.5625713554".
the Y of the a37r4 is "1399103.4828669566".

[create a53r4]
a53r4 is a area. "An area (53) in r4".
the printed name of the a53r4 is "Room 4".
Understand "Area 53 in Room 4" as a53r4.
the parent of the a53r4 is r4.
the X of the a53r4 is "547750.011235955".
the Y of the a53r4 is "1399000.0".

[create a54r4]
a54r4 is a area. "An area (54) in r4".
the printed name of the a54r4 is "Room 4".
Understand "Area 54 in Room 4" as a54r4.
the parent of the a54r4 is r4.
the X of the a54r4 is "547717.1427967965".
the Y of the a54r4 is "1398999.136272043".

[create a55r4]
a55r4 is a area. "An area (55) in r4".
the printed name of the a55r4 is "Room 4".
Understand "Area 55 in Room 4" as a55r4.
the parent of the a55r4 is r4.
the X of the a55r4 is "548473.0".
the Y of the a55r4 is "1399000.0".

[create a56r4]
a56r4 is a area. "An area (56) in r4".
the printed name of the a56r4 is "Room 4".
Understand "Area 56 in Room 4" as a56r4.
the parent of the a56r4 is r4.
the X of the a56r4 is "548437.0".
the Y of the a56r4 is "1399094.8699839485".

[create a59r4]
a59r4 is a area. "An area (59) in r4".
the printed name of the a59r4 is "Room 4".
Understand "Area 59 in Room 4" as a59r4.
the parent of the a59r4 is r4.
the X of the a59r4 is "548861.0".
the Y of the a59r4 is "1399356.0".

[create a60r4]
a60r4 is a area. "An area (60) in r4".
the printed name of the a60r4 is "Room 4".
Understand "Area 60 in Room 4" as a60r4.
the parent of the a60r4 is r4.
the X of the a60r4 is "548774.7060628375".
the Y of the a60r4 is "1399394.4166666667".

[create a61r4]
a61r4 is a area. "An area (61) in r4".
the printed name of the a61r4 is "Room 4".
Understand "Area 61 in Room 4" as a61r4.
the parent of the a61r4 is r4.
the X of the a61r4 is "548861.0".
the Y of the a61r4 is "1399817.0".

[create a62r4]
a62r4 is a area. "An area (62) in r4".
the printed name of the a62r4 is "Room 4".
Understand "Area 62 in Room 4" as a62r4.
the parent of the a62r4 is r4.
the X of the a62r4 is "548757.3488372092".
the Y of the a62r4 is "1399778.5833333335".

[create a71r4]
a71r4 is a area. "An area (71) in r4".
the printed name of the a71r4 is "Room 4".
Understand "Area 71 in Room 4" as a71r4.
the parent of the a71r4 is r4.
the X of the a71r4 is "546908.0".
the Y of the a71r4 is "1398991.0".

[create a72r4]
a72r4 is a area. "An area (72) in r4".
the printed name of the a72r4 is "Room 4".
Understand "Area 72 in Room 4" as a72r4.
the parent of the a72r4 is r4.
the X of the a72r4 is "546939.0234020578".
the Y of the a72r4 is "1399017.0166743887".

[create a75r4]
a75r4 is a area. "An area (75) in r4".
the printed name of the a75r4 is "Room 4".
Understand "Area 75 in Room 4" as a75r4.
the parent of the a75r4 is r4.
the X of the a75r4 is "546898.0".
the Y of the a75r4 is "1399080.0".

[create a76r4]
a76r4 is a area. "An area (76) in r4".
the printed name of the a76r4 is "Room 4".
Understand "Area 76 in Room 4" as a76r4.
the parent of the a76r4 is r4.
the X of the a76r4 is "546929.3505784649".
the Y of the a76r4 is "1399060.9916380297".

[create a81r4]
a81r4 is a area. "An area (81) in r4".
the printed name of the a81r4 is "Room 4".
Understand "Area 81 in Room 4" as a81r4.
the parent of the a81r4 is r4.
the X of the a81r4 is "547185.0898876404".
the Y of the a81r4 is "1399000.0".

[create a82r4]
a82r4 is a area. "An area (82) in r4".
the printed name of the a82r4 is "Room 4".
Understand "Area 82 in Room 4" as a82r4.
the parent of the a82r4 is r4.
the X of the a82r4 is "547218.0417059886".
the Y of the a82r4 is "1398998.786189433".

[create d0]
d0 is a door. "A door between a40r0 and a6r1".
d0 is north of a40r0 and south of a6r1.

[create d2]
d2 is a door. "A door between a27r2 and a3r3".
d2 is northeast of a27r2 and southwest of a3r3.

[create d4]
d4 is a door. "A door between a6r1 and a27r2".
d4 is southeast of a6r1 and northwest of a27r2.

[create d6]
d6 is a door. "A door between a16r3 and a14r4".
d6 is south of a16r3 and north of a14r4.

[create d7]
d7 is a door. "A door between a3r3 and a15r4".
d7 is south of a3r3 and north of a15r4.

[create d8]
d8 is a door. "A door between a18r3 and a26r4".
d8 is southeast of a18r3 and northwest of a26r4.

south of a0r3 is north of a1r3.

west of a0r3 is east of a2r3.

north of a3r3 is south of a4r3.

south of a5r1 is north of a6r1.

west of a5r1 is east of a7r1.

northeast of a5r1 is southwest of a8r1.

east of a9r3 is west of a10r3.

south of a11r4 is north of a12r4.

north of a13r4 is south of a14r4.

north of a11r4 is south of a15r4.

east of a17r3 is west of a18r3.

southwest of a1r3 is northeast of a18r3.

south of a10r3 is north of a17r3.

north of a19r4 is south of a20r4.

south of a21r4 is north of a22r4.

north of a20r4 is south of a23r4.

north of a21r4 is south of a24r4.

north of a23r4 is south of a25r4.

south of a12r4 is north of a24r4.

south of a13r4 is north of a25r4.

east of a28r4 is west of a29r4.

north of a28r4 is south of a30r4.

north of a29r4 is south of a31r4.

south of a26r4 is north of a32r4.

north of a30r4 is south of a33r4.

south of a19r4 is north of a31r4.

south of a22r4 is north of a33r4.

east of a34r4 is west of a35r4.

east of a36r4 is west of a37r4.

northeast of a38r2 is southwest of a39r2.

southeast of a29r4 is northwest of a36r4.

southwest of a32r4 is northeast of a37r4.

southwest of a28r4 is northeast of a35r4.

east of a40r0 is west of a41r0.

west of a40r0 is east of a42r0.

northeast of a43r1 is southwest of a44r1.

southeast of a45r3 is northwest of a46r3.

southeast of a47r1 is northwest of a48r1.

east of a49r3 is west of a50r3.

west of a51r3 is east of a52r3.

west of a53r4 is east of a54r4.

north of a55r4 is south of a56r4.

northwest of a57r2 is southeast of a58r2.

northwest of a59r4 is southeast of a60r4.

west of a61r4 is east of a62r4.

west of a63r3 is east of a64r3.

southwest of a65r3 is northeast of a66r3.

south of a67r3 is north of a68r3.

northwest of a69r0 is southeast of a70r0.

northeast of a71r4 is southwest of a72r4.

southwest of a73r0 is northeast of a74r0.

southeast of a75r4 is northwest of a76r4.

northeast of a77r0 is southwest of a78r0.

southeast of a79r0 is northwest of a80r0.

east of a81r4 is west of a82r4.

southwest of a38r2 is northeast of a83r2.

northeast of a8r1 is southwest of a84r1.

east of a26r4 is west of a62r4.

north of a16r3 is south of a17r3.

east of a1r3 is west of a64r3.

northeast of a0r3 is southwest of a66r3.

east of a9r3 is west of a68r3.

northeast of a4r3 is southwest of a9r3.

northwest of a4r3 is southeast of a85r3.

southwest of a2r3 is northeast of a10r3.

southwest of a7r1 is northeast of a44r1.

southwest of a42r0 is northeast of a78r0.

northwest of a42r0 is southeast of a80r0.

south of a35r4 is north of a82r4.

south of a36r4 is north of a54r4.

south of a37r4 is north of a56r4.

southeast of a39r2 is northwest of a58r2.

south of a27r2 is north of a39r2.

southeast of a32r4 is northwest of a60r4.

southeast of a41r0 is northwest of a70r0.

southwest of a34r4 is northeast of a72r4.

northeast of a41r0 is southwest of a74r0.

northwest of a34r4 is southeast of a76r4.

southeast of a46r3 is northwest of a85r3.

northwest of a7r1 is southeast of a48r1.

south of a50r3 is north of a85r3.

north of a2r3 is south of a52r3.

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
the player is in a47r1.

the orientation of the player is 6.
