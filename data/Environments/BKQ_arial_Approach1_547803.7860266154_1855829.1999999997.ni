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

[create a0r0]
a0r0 is a area. "An area (0) in r0".
the printed name of the a0r0 is "Room 0".
Understand "Area 0 in Room 0" as a0r0.
the parent of the a0r0 is r0.
the X of the a0r0 is "549664.6833326339".
the Y of the a0r0 is "1855002.5983136436".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "549704.7174918372".
the Y of the a1r0 is "1854917.9228117736".

[create a2r0]
a2r0 is a area. "An area (2) in r0".
the printed name of the a2r0 is "Room 0".
Understand "Area 2 in Room 0" as a2r0.
the parent of the a2r0 is r0.
the X of the a2r0 is "549766.628155661".
the Y of the a2r0 is "1855043.601250532".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "547810.658679734".
the Y of the a22r0 is "1855830.9579340303".

[create a27r0]
a27r0 is a area. "An area (27) in r0".
the printed name of the a27r0 is "Room 0".
Understand "Area 27 in Room 0" as a27r0.
the parent of the a27r0 is r0.
the X of the a27r0 is "548858.3451167858".
the Y of the a27r0 is "1855516.4676672716".

[create a28r0]
a28r0 is a area. "An area (28) in r0".
the printed name of the a28r0 is "Room 0".
Understand "Area 28 in Room 0" as a28r0.
the parent of the a28r0 is r0.
the X of the a28r0 is "549173.1766160899".
the Y of the a28r0 is "1855279.2736719858".

[create a29r0]
a29r0 is a area. "An area (29) in r0".
the printed name of the a29r0 is "Room 0".
Understand "Area 29 in Room 0" as a29r0.
the parent of the a29r0 is r0.
the X of the a29r0 is "549273.543451193".
the Y of the a29r0 is "1855292.2661425492".

[create a30r0]
a30r0 is a area. "An area (30) in r0".
the printed name of the a30r0 is "Room 0".
Understand "Area 30 in Room 0" as a30r0.
the parent of the a30r0 is r0.
the X of the a30r0 is "548003.5090763445".
the Y of the a30r0 is "1855445.7178598023".

[create a31r0]
a31r0 is a area. "An area (31) in r0".
the printed name of the a31r0 is "Room 0".
Understand "Area 31 in Room 0" as a31r0.
the parent of the a31r0 is r0.
the X of the a31r0 is "548041.990862943".
the Y of the a31r0 is "1855395.1593868812".

[create a32r0]
a32r0 is a area. "An area (32) in r0".
the printed name of the a32r0 is "Room 0".
Understand "Area 32 in Room 0" as a32r0.
the parent of the a32r0 is r0.
the X of the a32r0 is "548768.5987250463".
the Y of the a32r0 is "1855467.6289370495".

[create a67r0]
a67r0 is a area. "An area (67) in r0".
the printed name of the a67r0 is "Room 0".
Understand "Area 67 in Room 0" as a67r0.
the parent of the a67r0 is r0.
the X of the a67r0 is "548016.9569064742".
the Y of the a67r0 is "1855371.633121643".

[create a72r0]
a72r0 is a area. "An area (72) in r0".
the printed name of the a72r0 is "Room 0".
Understand "Area 72 in Room 0" as a72r0.
the parent of the a72r0 is r0.
the X of the a72r0 is "548793.0".
the Y of the a72r0 is "1855640.0".

[create a73r0]
a73r0 is a area. "An area (73) in r0".
the printed name of the a73r0 is "Room 0".
Understand "Area 73 in Room 0" as a73r0.
the parent of the a73r0 is r0.
the X of the a73r0 is "548814.9592895454".
the Y of the a73r0 is "1855595.7405367976".

[create a76r0]
a76r0 is a area. "An area (76) in r0".
the printed name of the a76r0 is "Room 0".
Understand "Area 76 in Room 0" as a76r0.
the parent of the a76r0 is r0.
the X of the a76r0 is "547662.0".
the Y of the a76r0 is "1855797.0".

[create a77r0]
a77r0 is a area. "An area (77) in r0".
the printed name of the a77r0 is "Room 0".
Understand "Area 77 in Room 0" as a77r0.
the parent of the a77r0 is r0.
the X of the a77r0 is "547803.7860266154".
the Y of the a77r0 is "1855829.1999999997".

[create a81r0]
a81r0 is a area. "An area (81) in r0".
the printed name of the a81r0 is "Room 0".
Understand "Area 81 in Room 0" as a81r0.
the parent of the a81r0 is r0.
the X of the a81r0 is "549722.0".
the Y of the a81r0 is "1854872.0".

[create a82r0]
a82r0 is a area. "An area (82) in r0".
the printed name of the a82r0 is "Room 0".
Understand "Area 82 in Room 0" as a82r0.
the parent of the a82r0 is r0.
the X of the a82r0 is "549807.0".
the Y of the a82r0 is "1855064.0".

[create a85r0]
a85r0 is a area. "An area (85) in r0".
the printed name of the a85r0 is "Room 0".
Understand "Area 85 in Room 0" as a85r0.
the parent of the a85r0 is r0.
the X of the a85r0 is "548723.0".
the Y of the a85r0 is "1855448.0".

[create a11r1]
a11r1 is a area. "An area (11) in r1".
the printed name of the a11r1 is "Room 1".
Understand "Area 11 in Room 1" as a11r1.
the parent of the a11r1 is r1.
the X of the a11r1 is "547810.3571895964".
the Y of the a11r1 is "1856407.1959411195".

[create a12r1]
a12r1 is a area. "An area (12) in r1".
the printed name of the a12r1 is "Room 1".
Understand "Area 12 in Room 1" as a12r1.
the parent of the a12r1 is r1.
the X of the a12r1 is "548043.4472689205".
the Y of the a12r1 is "1856852.134567215".

[create a13r1]
a13r1 is a area. "An area (13) in r1".
the printed name of the a13r1 is "Room 1".
Understand "Area 13 in Room 1" as a13r1.
the parent of the a13r1 is r1.
the X of the a13r1 is "549401.4754669056".
the Y of the a13r1 is "1856853.915743845".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "549615.4210697461".
the Y of the a14r1 is "1856409.5093008713".

[create a50r1]
a50r1 is a area. "An area (50) in r1".
the printed name of the a50r1 is "Room 1".
Understand "Area 50 in Room 1" as a50r1.
the parent of the a50r1 is r1.
the X of the a50r1 is "547662.0".
the Y of the a50r1 is "1856440.0".

[create a51r1]
a51r1 is a area. "An area (51) in r1".
the printed name of the a51r1 is "Room 1".
Understand "Area 51 in Room 1" as a51r1.
the parent of the a51r1 is r1.
the X of the a51r1 is "547807.5873579548".
the Y of the a51r1 is "1856407.9".

[create a52r1]
a52r1 is a area. "An area (52) in r1".
the printed name of the a52r1 is "Room 1".
Understand "Area 52 in Room 1" as a52r1.
the parent of the a52r1 is r1.
the X of the a52r1 is "549765.0".
the Y of the a52r1 is "1856443.0".

[create a53r1]
a53r1 is a area. "An area (53) in r1".
the printed name of the a53r1 is "Room 1".
Understand "Area 53 in Room 1" as a53r1.
the parent of the a53r1 is r1.
the X of the a53r1 is "549620.6000000001".
the Y of the a53r1 is "1856410.8".

[create a60r1]
a60r1 is a area. "An area (60) in r1".
the printed name of the a60r1 is "Room 1".
Understand "Area 60 in Room 1" as a60r1.
the parent of the a60r1 is r1.
the X of the a60r1 is "549509.0".
the Y of the a60r1 is "1856955.0".

[create a61r1]
a61r1 is a area. "An area (61) in r1".
the printed name of the a61r1 is "Room 1".
Understand "Area 61 in Room 1" as a61r1.
the parent of the a61r1 is r1.
the X of the a61r1 is "549418.3170995669".
the Y of the a61r1 is "1856869.6585497835".

[create a62r1]
a62r1 is a area. "An area (62) in r1".
the printed name of the a62r1 is "Room 1".
Understand "Area 62 in Room 1" as a62r1.
the parent of the a62r1 is r1.
the X of the a62r1 is "547926.0".
the Y of the a62r1 is "1856965.0".

[create a63r1]
a63r1 is a area. "An area (63) in r1".
the printed name of the a63r1 is "Room 1".
Understand "Area 63 in Room 1" as a63r1.
the parent of the a63r1 is r1.
the X of the a63r1 is "548014.4027777776".
the Y of the a63r1 is "1856879.4363888893".

[create a5r2]
a5r2 is a area. "An area (5) in r2".
the printed name of the a5r2 is "Room 2".
Understand "Area 5 in Room 2" as a5r2.
the parent of the a5r2 is r2.
the X of the a5r2 is "546411.1564648793".
the Y of the a5r2 is "1856159.3623601177".

[create a6r2]
a6r2 is a area. "An area (6) in r2".
the printed name of the a6r2 is "Room 2".
Understand "Area 6 in Room 2" as a6r2.
the parent of the a6r2 is r2.
the X of the a6r2 is "546494.4629540613".
the Y of the a6r2 is "1856394.2123620857".

[create a7r2]
a7r2 is a area. "An area (7) in r2".
the printed name of the a7r2 is "Room 2".
Understand "Area 7 in Room 2" as a7r2.
the parent of the a7r2 is r2.
the X of the a7r2 is "545386.016849386".
the Y of the a7r2 is "1857079.2377959513".

[create a8r2]
a8r2 is a area. "An area (8) in r2".
the printed name of the a8r2 is "Room 2".
Understand "Area 8 in Room 2" as a8r2.
the parent of the a8r2 is r2.
the X of the a8r2 is "545398.575887403".
the Y of the a8r2 is "1856152.0".

[create a9r2]
a9r2 is a area. "An area (9) in r2".
the printed name of the a9r2 is "Room 2".
Understand "Area 9 in Room 2" as a9r2.
the parent of the a9r2 is r2.
the X of the a9r2 is "547002.2981886293".
the Y of the a9r2 is "1856291.9572795876".

[create a10r2]
a10r2 is a area. "An area (10) in r2".
the printed name of the a10r2 is "Room 2".
Understand "Area 10 in Room 2" as a10r2.
the parent of the a10r2 is r2.
the X of the a10r2 is "547751.1710420791".
the Y of the a10r2 is "1857100.735986112".

[create a15r2]
a15r2 is a area. "An area (15) in r2".
the printed name of the a15r2 is "Room 2".
Understand "Area 15 in Room 2" as a15r2.
the parent of the a15r2 is r2.
the X of the a15r2 is "546543.4992936216".
the Y of the a15r2 is "1856577.411465467".

[create a16r2]
a16r2 is a area. "An area (16) in r2".
the printed name of the a16r2 is "Room 2".
Understand "Area 16 in Room 2" as a16r2.
the parent of the a16r2 is r2.
the X of the a16r2 is "546542.6065875137".
the Y of the a16r2 is "1856748.9661634325".

[create a17r2]
a17r2 is a area. "An area (17) in r2".
the printed name of the a17r2 is "Room 2".
Understand "Area 17 in Room 2" as a17r2.
the parent of the a17r2 is r2.
the X of the a17r2 is "546480.0455859327".
the Y of the a17r2 is "1856978.740950441".

[create a38r2]
a38r2 is a area. "An area (38) in r2".
the printed name of the a38r2 is "Room 2".
Understand "Area 38 in Room 2" as a38r2.
the parent of the a38r2 is r2.
the X of the a38r2 is "546267.8879337106".
the Y of the a38r2 is "1857103.3308703958".

[create a39r2]
a39r2 is a area. "An area (39) in r2".
the printed name of the a39r2 is "Room 2".
Understand "Area 39 in Room 2" as a39r2.
the parent of the a39r2 is r2.
the X of the a39r2 is "546351.789241452".
the Y of the a39r2 is "1857091.083060109".

[create a40r2]
a40r2 is a area. "An area (40) in r2".
the printed name of the a40r2 is "Room 2".
Understand "Area 40 in Room 2" as a40r2.
the parent of the a40r2 is r2.
the X of the a40r2 is "546430.7642810689".
the Y of the a40r2 is "1857068.4644792383".

[create a41r2]
a41r2 is a area. "An area (41) in r2".
the printed name of the a41r2 is "Room 2".
Understand "Area 41 in Room 2" as a41r2.
the parent of the a41r2 is r2.
the X of the a41r2 is "545262.4999999998".
the Y of the a41r2 is "1857201.3793103448".

[create a42r2]
a42r2 is a area. "An area (42) in r2".
the printed name of the a42r2 is "Room 2".
Understand "Area 42 in Room 2" as a42r2.
the parent of the a42r2 is r2.
the X of the a42r2 is "546563.6250000002".
the Y of the a42r2 is "1857206.291803279".

[create a43r2]
a43r2 is a area. "An area (43) in r2".
the printed name of the a43r2 is "Room 2".
Understand "Area 43 in Room 2" as a43r2.
the parent of the a43r2 is r2.
the X of the a43r2 is "548047.4999999998".
the Y of the a43r2 is "1857225.6353005604".

[create a44r2]
a44r2 is a area. "An area (44) in r2".
the printed name of the a44r2 is "Room 2".
Understand "Area 44 in Room 2" as a44r2.
the parent of the a44r2 is r2.
the X of the a44r2 is "546302.0".
the Y of the a44r2 is "1857240.0".

[create a45r2]
a45r2 is a area. "An area (45) in r2".
the printed name of the a45r2 is "Room 2".
Understand "Area 45 in Room 2" as a45r2.
the parent of the a45r2 is r2.
the X of the a45r2 is "546270.5".
the Y of the a45r2 is "1857161.0613209773".

[create a46r2]
a46r2 is a area. "An area (46) in r2".
the printed name of the a46r2 is "Room 2".
Understand "Area 46 in Room 2" as a46r2.
the parent of the a46r2 is r2.
the X of the a46r2 is "547692.0".
the Y of the a46r2 is "1857240.0".

[create a47r2]
a47r2 is a area. "An area (47) in r2".
the printed name of the a47r2 is "Room 2".
Understand "Area 47 in Room 2" as a47r2.
the parent of the a47r2 is r2.
the X of the a47r2 is "547731.5000000002".
the Y of the a47r2 is "1857147.9035564975".

[create a48r2]
a48r2 is a area. "An area (48) in r2".
the printed name of the a48r2 is "Room 2".
Understand "Area 48 in Room 2" as a48r2.
the parent of the a48r2 is r2.
the X of the a48r2 is "546309.0".
the Y of the a48r2 is "1857240.0".

[create a49r2]
a49r2 is a area. "An area (49) in r2".
the printed name of the a49r2 is "Room 2".
Understand "Area 49 in Room 2" as a49r2.
the parent of the a49r2 is r2.
the X of the a49r2 is "546345.375".
the Y of the a49r2 is "1857238.4163934428".

[create a54r2]
a54r2 is a area. "An area (54) in r2".
the printed name of the a54r2 is "Room 2".
Understand "Area 54 in Room 2" as a54r2.
the parent of the a54r2 is r2.
the X of the a54r2 is "546600.0".
the Y of the a54r2 is "1856241.6623376624".

[create a55r2]
a55r2 is a area. "An area (55) in r2".
the printed name of the a55r2 is "Room 2".
Understand "Area 55 in Room 2" as a55r2.
the parent of the a55r2 is r2.
the X of the a55r2 is "546570.2603186157".
the Y of the a55r2 is "1856331.6464970997".

[create a56r2]
a56r2 is a area. "An area (56) in r2".
the printed name of the a56r2 is "Room 2".
Understand "Area 56 in Room 2" as a56r2.
the parent of the a56r2 is r2.
the X of the a56r2 is "546779.0".
the Y of the a56r2 is "1856513.0".

[create a57r2]
a57r2 is a area. "An area (57) in r2".
the printed name of the a57r2 is "Room 2".
Understand "Area 57 in Room 2" as a57r2.
the parent of the a57r2 is r2.
the X of the a57r2 is "546644.3651960786".
the Y of the a57r2 is "1856549.2500000002".

[create a58r2]
a58r2 is a area. "An area (58) in r2".
the printed name of the a58r2 is "Room 2".
Understand "Area 58 in Room 2" as a58r2.
the parent of the a58r2 is r2.
the X of the a58r2 is "546779.0".
the Y of the a58r2 is "1856815.0".

[create a59r2]
a59r2 is a area. "An area (59) in r2".
the printed name of the a59r2 is "Room 2".
Understand "Area 59 in Room 2" as a59r2.
the parent of the a59r2 is r2.
the X of the a59r2 is "546648.418313954".
the Y of the a59r2 is "1856775.75".

[create a64r2]
a64r2 is a area. "An area (64) in r2".
the printed name of the a64r2 is "Room 2".
Understand "Area 64 in Room 2" as a64r2.
the parent of the a64r2 is r2.
the X of the a64r2 is "546607.0".
the Y of the a64r2 is "1857102.0".

[create a65r2]
a65r2 is a area. "An area (65) in r2".
the printed name of the a65r2 is "Room 2".
Understand "Area 65 in Room 2" as a65r2.
the parent of the a65r2 is r2.
the X of the a65r2 is "546576.9988366391".
the Y of the a65r2 is "1857045.01219478".

[create a78r2]
a78r2 is a area. "An area (78) in r2".
the printed name of the a78r2 is "Room 2".
Understand "Area 78 in Room 2" as a78r2.
the parent of the a78r2 is r2.
the X of the a78r2 is "545225.0".
the Y of the a78r2 is "1857240.0".

[create a79r2]
a79r2 is a area. "An area (79) in r2".
the printed name of the a79r2 is "Room 2".
Understand "Area 79 in Room 2" as a79r2.
the parent of the a79r2 is r2.
the X of the a79r2 is "546600.0".
the Y of the a79r2 is "1857240.0".

[create a80r2]
a80r2 is a area. "An area (80) in r2".
the printed name of the a80r2 is "Room 2".
Understand "Area 80 in Room 2" as a80r2.
the parent of the a80r2 is r2.
the X of the a80r2 is "548087.0".
the Y of the a80r2 is "1857240.0".

[create a3r3]
a3r3 is a area. "An area (3) in r3".
the printed name of the a3r3 is "Room 3".
Understand "Area 3 in Room 3" as a3r3.
the parent of the a3r3 is r3.
the X of the a3r3 is "547854.339182965".
the Y of the a3r3 is "1855139.7718913804".

[create a4r3]
a4r3 is a area. "An area (4) in r3".
the printed name of the a4r3 is "Room 3".
Understand "Area 4 in Room 3" as a4r3.
the parent of the a4r3 is r3.
the X of the a4r3 is "548088.6".
the Y of the a4r3 is "1855020.3676678508".

[create a24r3]
a24r3 is a area. "An area (24) in r3".
the printed name of the a24r3 is "Room 3".
Understand "Area 24 in Room 3" as a24r3.
the parent of the a24r3 is r3.
the X of the a24r3 is "547707.2363009846".
the Y of the a24r3 is "1855322.426372225".

[create a25r3]
a25r3 is a area. "An area (25) in r3".
the printed name of the a25r3 is "Room 3".
Understand "Area 25 in Room 3" as a25r3.
the parent of the a25r3 is r3.
the X of the a25r3 is "547752.2796075288".
the Y of the a25r3 is "1855258.6452974153".

[create a37r3]
a37r3 is a area. "An area (37) in r3".
the printed name of the a37r3 is "Room 3".
Understand "Area 37 in Room 3" as a37r3.
the parent of the a37r3 is r3.
the X of the a37r3 is "547824.0369852055".
the Y of the a37r3 is "1855154.6101491826".

[create a66r3]
a66r3 is a area. "An area (66) in r3".
the printed name of the a66r3 is "Room 3".
Understand "Area 66 in Room 3" as a66r3.
the parent of the a66r3 is r3.
the X of the a66r3 is "548127.0".
the Y of the a66r3 is "1855000.0".

[create a87r3]
a87r3 is a area. "An area (87) in r3".
the printed name of the a87r3 is "Room 3".
Understand "Area 87 in Room 3" as a87r3.
the parent of the a87r3 is r3.
the X of the a87r3 is "547781.4".
the Y of the a87r3 is "1855073.0499656457".

[create a18r4]
a18r4 is a area. "An area (18) in r4".
the printed name of the a18r4 is "Room 4".
Understand "Area 18 in Room 4" as a18r4.
the parent of the a18r4 is r4.
the X of the a18r4 is "546557.077919824".
the Y of the a18r4 is "1855383.5139295058".

[create a19r4]
a19r4 is a area. "An area (19) in r4".
the printed name of the a19r4 is "Room 4".
Understand "Area 19 in Room 4" as a19r4.
the parent of the a19r4 is r4.
the X of the a19r4 is "546597.461628628".
the Y of the a19r4 is "1855552.434085899".

[create a20r4]
a20r4 is a area. "An area (20) in r4".
the printed name of the a20r4 is "Room 4".
Understand "Area 20 in Room 4" as a20r4.
the parent of the a20r4 is r4.
the X of the a20r4 is "546597.70390252".
the Y of the a20r4 is "1855734.7395494324".

[create a21r4]
a21r4 is a area. "An area (21) in r4".
the printed name of the a21r4 is "Room 4".
Understand "Area 21 in Room 4" as a21r4.
the parent of the a21r4 is r4.
the X of the a21r4 is "546799.9874900229".
the Y of the a21r4 is "1855801.8333333335".

[create a23r4]
a23r4 is a area. "An area (23) in r4".
the printed name of the a23r4 is "Room 4".
Understand "Area 23 in Room 4" as a23r4.
the parent of the a23r4 is r4.
the X of the a23r4 is "546552.881982254".
the Y of the a23r4 is "1855895.6522984328".

[create a26r4]
a26r4 is a area. "An area (26) in r4".
the printed name of the a26r4 is "Room 4".
Understand "Area 26 in Room 4" as a26r4.
the parent of the a26r4 is r4.
the X of the a26r4 is "546443.5811062193".
the Y of the a26r4 is "1855170.38964552".

[create a33r4]
a33r4 is a area. "An area (33) in r4".
the printed name of the a33r4 is "Room 4".
Understand "Area 33 in Room 4" as a33r4.
the parent of the a33r4 is r4.
the X of the a33r4 is "546563.625".
the Y of the a33r4 is "1855025.4856770835".

[create a34r4]
a34r4 is a area. "An area (34) in r4".
the printed name of the a34r4 is "Room 4".
Understand "Area 34 in Room 4" as a34r4.
the parent of the a34r4 is r4.
the X of the a34r4 is "545263.1363636364".
the Y of the a34r4 is "1855038.6206896554".

[create a35r4]
a35r4 is a area. "An area (35) in r4".
the printed name of the a35r4 is "Room 4".
Understand "Area 35 in Room 4" as a35r4.
the parent of the a35r4 is r4.
the X of the a35r4 is "545385.9751893163".
the Y of the a35r4 is "1855160.8816229785".

[create a36r4]
a36r4 is a area. "An area (36) in r4".
the printed name of the a36r4 is "Room 4".
Understand "Area 36 in Room 4" as a36r4.
the parent of the a36r4 is r4.
the X of the a36r4 is "546362.3417035278".
the Y of the a36r4 is "1855142.210968858".

[create a68r4]
a68r4 is a area. "An area (68) in r4".
the printed name of the a68r4 is "Room 4".
Understand "Area 68 in Room 4" as a68r4.
the parent of the a68r4 is r4.
the X of the a68r4 is "546309.0".
the Y of the a68r4 is "1855000.0".

[create a69r4]
a69r4 is a area. "An area (69) in r4".
the printed name of the a69r4 is "Room 4".
Understand "Area 69 in Room 4" as a69r4.
the parent of the a69r4 is r4.
the X of the a69r4 is "546345.375".
the Y of the a69r4 is "1855004.480968858".

[create a70r4]
a70r4 is a area. "An area (70) in r4".
the printed name of the a70r4 is "Room 4".
Understand "Area 70 in Room 4" as a70r4.
the parent of the a70r4 is r4.
the X of the a70r4 is "546755.0".
the Y of the a70r4 is "1855219.0".

[create a71r4]
a71r4 is a area. "An area (71) in r4".
the printed name of the a71r4 is "Room 4".
Understand "Area 71 in Room 4" as a71r4.
the parent of the a71r4 is r4.
the X of the a71r4 is "546640.7331136041".
the Y of the a71r4 is "1855316.588673375".

[create a74r4]
a74r4 is a area. "An area (74) in r4".
the printed name of the a74r4 is "Room 4".
Understand "Area 74 in Room 4" as a74r4.
the parent of the a74r4 is r4.
the X of the a74r4 is "546886.0".
the Y of the a74r4 is "1855485.0".

[create a75r4]
a75r4 is a area. "An area (75) in r4".
the printed name of the a75r4 is "Room 4".
Understand "Area 75 in Room 4" as a75r4.
the parent of the a75r4 is r4.
the X of the a75r4 is "546722.9188931299".
the Y of the a75r4 is "1855524.0".

[create a83r4]
a83r4 is a area. "An area (83) in r4".
the printed name of the a83r4 is "Room 4".
Understand "Area 83 in Room 4" as a83r4.
the parent of the a83r4 is r4.
the X of the a83r4 is "546600.0".
the Y of the a83r4 is "1855000.0".

[create a84r4]
a84r4 is a area. "An area (84) in r4".
the printed name of the a84r4 is "Room 4".
Understand "Area 84 in Room 4" as a84r4.
the parent of the a84r4 is r4.
the X of the a84r4 is "545225.0".
the Y of the a84r4 is "1855000.0".

[create a86r4]
a86r4 is a area. "An area (86) in r4".
the printed name of the a86r4 is "Room 4".
Understand "Area 86 in Room 4" as a86r4.
the parent of the a86r4 is r4.
the X of the a86r4 is "546886.0".
the Y of the a86r4 is "1855834.0".

[create d0]
d0 is a door. "A door between a5r2 and a23r4".
d0 is southeast of a5r2 and northwest of a23r4.

[create d1]
d1 is a door. "A door between a9r2 and a23r4".
d1 is southwest of a9r2 and northeast of a23r4.

[create d2]
d2 is a door. "A door between a22r0 and a11r1".
d2 is north of a22r0 and south of a11r1.

[create d3]
d3 is a door. "A door between a29r0 and a14r1".
d3 is north of a29r0 and south of a14r1.

[create d4]
d4 is a door. "A door between a9r2 and a24r3".
d4 is southeast of a9r2 and northwest of a24r3.

[create d8]
d8 is a door. "A door between a8r2 and a35r4".
d8 is south of a8r2 and north of a35r4.

southeast of a0r0 is northwest of a1r0.

east of a0r0 is west of a2r0.

southeast of a3r3 is northwest of a4r3.

north of a5r2 is south of a6r2.

south of a7r2 is north of a8r2.

northeast of a9r2 is southwest of a10r2.

northeast of a11r1 is southwest of a12r1.

southeast of a13r1 is northwest of a14r1.

north of a6r2 is south of a15r2.

north of a15r2 is south of a16r2.

north of a16r2 is south of a17r2.

north of a18r4 is south of a19r4.

north of a19r4 is south of a20r4.

east of a20r4 is west of a21r4.

north of a20r4 is south of a23r4.

west of a5r2 is east of a8r2.

southeast of a24r3 is northwest of a25r3.

southwest of a18r4 is northeast of a26r4.

southeast of a27r0 is northwest of a28r0.

southeast of a30r0 is northwest of a31r0.

southwest of a27r0 is northeast of a32r0.

southeast of a22r0 is northwest of a30r0.

northwest of a0r0 is southeast of a29r0.

southeast of a26r4 is northwest of a33r4.

northeast of a34r4 is southwest of a35r4.

west of a28r0 is east of a31r0.

east of a35r4 is west of a36r4.

west of a26r4 is east of a36r4.

southeast of a25r3 is northwest of a37r3.

east of a12r1 is west of a13r1.

east of a38r2 is west of a39r2.

east of a7r2 is west of a38r2.

east of a39r2 is west of a40r2.

northwest of a7r2 is southeast of a41r2.

northeast of a40r2 is southwest of a42r2.

northeast of a10r2 is southwest of a43r2.

south of a44r2 is north of a45r2.

southeast of a46r2 is northwest of a47r2.

east of a48r2 is west of a49r2.

east of a50r1 is west of a51r1.

west of a52r1 is east of a53r1.

north of a54r2 is south of a55r2.

west of a56r2 is east of a57r2.

west of a58r2 is east of a59r2.

southwest of a60r1 is northeast of a61r1.

southeast of a62r1 is northwest of a63r1.

southwest of a64r2 is northeast of a65r2.

east of a68r4 is west of a69r4.

northwest of a70r4 is southeast of a71r4.

southeast of a72r0 is northwest of a73r0.

west of a74r4 is east of a75r4.

east of a76r0 is west of a77r0.

northwest of a41r2 is southeast of a78r2.

northeast of a42r2 is southwest of a79r2.

east of a43r2 is west of a80r2.

south of a1r0 is north of a81r0.

northeast of a2r0 is southwest of a82r0.

southeast of a4r3 is northwest of a66r3.

southeast of a33r4 is northwest of a83r4.

southwest of a34r4 is northeast of a84r4.

southwest of a32r0 is northeast of a85r0.

east of a21r4 is west of a86r4.

northwest of a3r3 is southeast of a37r3.

east of a28r0 is west of a29r0.

southwest of a31r0 is northeast of a67r0.

southeast of a18r4 is northwest of a71r4.

northwest of a27r0 is southeast of a73r0.

east of a19r4 is west of a75r4.

west of a22r0 is east of a77r0.

west of a11r1 is east of a51r1.

east of a14r1 is west of a53r1.

south of a36r4 is north of a69r4.

southwest of a37r3 is northeast of a87r3.

north of a38r2 is south of a45r2.

northwest of a10r2 is southeast of a47r2.

north of a39r2 is south of a49r2.

southeast of a6r2 is northwest of a55r2.

east of a15r2 is west of a57r2.

east of a16r2 is west of a59r2.

northeast of a13r1 is southwest of a61r1.

northwest of a12r1 is southeast of a63r1.

northeast of a17r2 is southwest of a65r2.

northwest of a17r2 is southeast of a40r2.

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
the player is in a77r0.

the orientation of the player is 6.
