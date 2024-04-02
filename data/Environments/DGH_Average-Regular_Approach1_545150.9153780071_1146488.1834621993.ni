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
the X of the a0r0 is "545098.0".
the Y of the a0r0 is "1144391.0".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "545129.2137161086".
the Y of the a1r0 is "1144416.6339712918".

[create a2r0]
a2r0 is a area. "An area (2) in r0".
the printed name of the a2r0 is "Room 0".
Understand "Area 2 in Room 0" as a2r0.
the parent of the a2r0 is r0.
the X of the a2r0 is "549560.0".
the Y of the a2r0 is "1144391.0".

[create a3r0]
a3r0 is a area. "An area (3) in r0".
the printed name of the a3r0 is "Room 0".
Understand "Area 3 in Room 0" as a3r0.
the parent of the a3r0 is r0.
the X of the a3r0 is "549591.2137161084".
the Y of the a3r0 is "1144416.633971292".

[create a4r0]
a4r0 is a area. "An area (4) in r0".
the printed name of the a4r0 is "Room 0".
Understand "Area 4 in Room 0" as a4r0.
the parent of the a4r0 is r0.
the X of the a4r0 is "545089.0".
the Y of the a4r0 is "1144480.0".

[create a5r0]
a5r0 is a area. "An area (5) in r0".
the printed name of the a5r0 is "Room 0".
Understand "Area 5 in Room 0" as a5r0.
the parent of the a5r0 is r0.
the X of the a5r0 is "545120.0059757696".
the Y of the a5r0 is "1144460.657907662".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "549551.0".
the Y of the a6r0 is "1144480.0".

[create a7r0]
a7r0 is a area. "An area (7) in r0".
the printed name of the a7r0 is "Room 0".
Understand "Area 7 in Room 0" as a7r0.
the parent of the a7r0 is r0.
the X of the a7r0 is "549582.0059757691".
the Y of the a7r0 is "1144460.657907662".

[create a8r0]
a8r0 is a area. "An area (8) in r0".
the printed name of the a8r0 is "Room 0".
Understand "Area 8 in Room 0" as a8r0.
the parent of the a8r0 is r0.
the X of the a8r0 is "548986.0".
the Y of the a8r0 is "1144391.0".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "548954.7862838916".
the Y of the a9r0 is "1144416.6339712918".

[create a22r0]
a22r0 is a area. "An area (22) in r0".
the printed name of the a22r0 is "Room 0".
Understand "Area 22 in Room 0" as a22r0.
the parent of the a22r0 is r0.
the X of the a22r0 is "548056.0".
the Y of the a22r0 is "1145343.0".

[create a23r0]
a23r0 is a area. "An area (23) in r0".
the printed name of the a23r0 is "Room 0".
Understand "Area 23 in Room 0" as a23r0.
the parent of the a23r0 is r0.
the X of the a23r0 is "548087.0420577896".
the Y of the a23r0 is "1145317.3833874145".

[create a24r0]
a24r0 is a area. "An area (24) in r0".
the printed name of the a24r0 is "Room 0".
Understand "Area 24 in Room 0" as a24r0.
the parent of the a24r0 is r0.
the X of the a24r0 is "548784.0".
the Y of the a24r0 is "1145343.0".

[create a25r0]
a25r0 is a area. "An area (25) in r0".
the printed name of the a25r0 is "Room 0".
Understand "Area 25 in Room 0" as a25r0.
the parent of the a25r0 is r0.
the X of the a25r0 is "548748.1314110584".
the Y of the a25r0 is "1145316.8953112306".

[create a38r0]
a38r0 is a area. "An area (38) in r0".
the printed name of the a38r0 is "Room 0".
Understand "Area 38 in Room 0" as a38r0.
the parent of the a38r0 is r0.
the X of the a38r0 is "550402.0".
the Y of the a38r0 is "1144391.0".

[create a39r0]
a39r0 is a area. "An area (39) in r0".
the printed name of the a39r0 is "Room 0".
Understand "Area 39 in Room 0" as a39r0.
the parent of the a39r0 is r0.
the X of the a39r0 is "550370.9579422104".
the Y of the a39r0 is "1144416.6166125855".

[create a40r0]
a40r0 is a area. "An area (40) in r0".
the printed name of the a40r0 is "Room 0".
Understand "Area 40 in Room 0" as a40r0.
the parent of the a40r0 is r0.
the X of the a40r0 is "548995.0".
the Y of the a40r0 is "1144480.0".

[create a41r0]
a41r0 is a area. "An area (41) in r0".
the printed name of the a41r0 is "Room 0".
Understand "Area 41 in Room 0" as a41r0.
the parent of the a41r0 is r0.
the X of the a41r0 is "548952.1485252036".
the Y of the a41r0 is "1144461.855767114".

[create a42r0]
a42r0 is a area. "An area (42) in r0".
the printed name of the a42r0 is "Room 0".
Understand "Area 42 in Room 0" as a42r0.
the parent of the a42r0 is r0.
the X of the a42r0 is "550411.0".
the Y of the a42r0 is "1144480.0".

[create a43r0]
a43r0 is a area. "An area (43) in r0".
the printed name of the a43r0 is "Room 0".
Understand "Area 43 in Room 0" as a43r0.
the parent of the a43r0 is r0.
the X of the a43r0 is "550368.1485252036".
the Y of the a43r0 is "1144461.8557671143".

[create a44r0]
a44r0 is a area. "An area (44) in r0".
the printed name of the a44r0 is "Room 0".
Understand "Area 44 in Room 0" as a44r0.
the parent of the a44r0 is r0.
the X of the a44r0 is "547260.0".
the Y of the a44r0 is "1144363.0".

[create a45r0]
a45r0 is a area. "An area (45) in r0".
the printed name of the a45r0 is "Room 0".
Understand "Area 45 in Room 0" as a45r0.
the parent of the a45r0 is r0.
the X of the a45r0 is "547296.0".
the Y of the a45r0 is "1144453.6590704143".

[create a46r0]
a46r0 is a area. "An area (46) in r0".
the printed name of the a46r0 is "Room 0".
Understand "Area 46 in Room 0" as a46r0.
the parent of the a46r0 is r0.
the X of the a46r0 is "546604.0".
the Y of the a46r0 is "1144400.0".

[create a47r0]
a47r0 is a area. "An area (47) in r0".
the printed name of the a47r0 is "Room 0".
Understand "Area 47 in Room 0" as a47r0.
the parent of the a47r0 is r0.
the X of the a47r0 is "546567.875".
the Y of the a47r0 is "1144483.8216703043".

[create a48r0]
a48r0 is a area. "An area (48) in r0".
the printed name of the a48r0 is "Room 0".
Understand "Area 48 in Room 0" as a48r0.
the parent of the a48r0 is r0.
the X of the a48r0 is "546720.8318446956".
the Y of the a48r0 is "1144865.6143820635".

[create a49r0]
a49r0 is a area. "An area (49) in r0".
the printed name of the a49r0 is "Room 0".
Understand "Area 49 in Room 0" as a49r0.
the parent of the a49r0 is r0.
the X of the a49r0 is "546751.6610348243".
the Y of the a49r0 is "1144949.642857143".

[create a50r0]
a50r0 is a area. "An area (50) in r0".
the printed name of the a50r0 is "Room 0".
Understand "Area 50 in Room 0" as a50r0.
the parent of the a50r0 is r0.
the X of the a50r0 is "548047.0".
the Y of the a50r0 is "1145254.0".

[create a51r0]
a51r0 is a area. "An area (51) in r0".
the printed name of the a51r0 is "Room 0".
Understand "Area 51 in Room 0" as a51r0.
the parent of the a51r0 is r0.
the X of the a51r0 is "548089.2240682532".
the Y of the a51r0 is "1145272.207678491".

[create a52r0]
a52r0 is a area. "An area (52) in r0".
the printed name of the a52r0 is "Room 0".
Understand "Area 52 in Room 0" as a52r0.
the parent of the a52r0 is r0.
the X of the a52r0 is "548747.0".
the Y of the a52r0 is "1145245.0".

[create a53r0]
a53r0 is a area. "An area (53) in r0".
the printed name of the a53r0 is "Room 0".
Understand "Area 53 in Room 0" as a53r0.
the parent of the a53r0 is r0.
the X of the a53r0 is "548765.2704191617".
the Y of the a53r0 is "1145273.6734131735".

[create a54r0]
a54r0 is a area. "An area (54) in r0".
the printed name of the a54r0 is "Room 0".
Understand "Area 54 in Room 0" as a54r0.
the parent of the a54r0 is r0.
the X of the a54r0 is "547212.0".
the Y of the a54r0 is "1145027.0".

[create a55r0]
a55r0 is a area. "An area (55) in r0".
the printed name of the a55r0 is "Room 0".
Understand "Area 55 in Room 0" as a55r0.
the parent of the a55r0 is r0.
the X of the a55r0 is "547122.3327468365".
the Y of the a55r0 is "1145062.2857142857".

[create a68r0]
a68r0 is a area. "An area (68) in r0".
the printed name of the a68r0 is "Room 0".
Understand "Area 68 in Room 0" as a68r0.
the parent of the a68r0 is r0.
the X of the a68r0 is "548104.0200662519".
the Y of the a68r0 is "1145289.2230761896".

[create a69r0]
a69r0 is a area. "An area (69) in r0".
the printed name of the a69r0 is "Room 0".
Understand "Area 69 in Room 0" as a69r0.
the parent of the a69r0 is r0.
the X of the a69r0 is "548736.9945280896".
the Y of the a69r0 is "1145295.6070483045".

[create a73r0]
a73r0 is a area. "An area (73) in r0".
the printed name of the a73r0 is "Room 0".
Understand "Area 73 in Room 0" as a73r0.
the parent of the a73r0 is r0.
the X of the a73r0 is "548937.7817693063".
the Y of the a73r0 is "1144444.9748289345".

[create a74r0]
a74r0 is a area. "An area (74) in r0".
the printed name of the a74r0 is "Room 0".
Understand "Area 74 in Room 0" as a74r0.
the parent of the a74r0 is r0.
the X of the a74r0 is "550353.8279210808".
the Y of the a74r0 is "1144445.02905727".

[create a75r0]
a75r0 is a area. "An area (75) in r0".
the printed name of the a75r0 is "Room 0".
Understand "Area 75 in Room 0" as a75r0.
the parent of the a75r0 is r0.
the X of the a75r0 is "547313.6269151601".
the Y of the a75r0 is "1144498.7933835285".

[create a76r0]
a76r0 is a area. "An area (76) in r0".
the printed name of the a76r0 is "Room 0".
Understand "Area 76 in Room 0" as a76r0.
the parent of the a76r0 is r0.
the X of the a76r0 is "546546.1199164822".
the Y of the a76r0 is "1144536.1709066466".

[create a77r0]
a77r0 is a area. "An area (77) in r0".
the printed name of the a77r0 is "Room 0".
Understand "Area 77 in Room 0" as a77r0.
the parent of the a77r0 is r0.
the X of the a77r0 is "546898.045670187".
the Y of the a77r0 is "1145011.3062275022".

[create a78r0]
a78r0 is a area. "An area (78) in r0".
the printed name of the a78r0 is "Room 0".
Understand "Area 78 in Room 0" as a78r0.
the parent of the a78r0 is r0.
the X of the a78r0 is "546914.2356786645".
the Y of the a78r0 is "1144896.8727861627".

[create a79r0]
a79r0 is a area. "An area (79) in r0".
the printed name of the a79r0 is "Room 0".
Understand "Area 79 in Room 0" as a79r0.
the parent of the a79r0 is r0.
the X of the a79r0 is "545143.389199255".
the Y of the a79r0 is "1144440.2597765364".

[create a80r0]
a80r0 is a area. "An area (80) in r0".
the printed name of the a80r0 is "Room 0".
Understand "Area 80 in Room 0" as a80r0.
the parent of the a80r0 is r0.
the X of the a80r0 is "549605.389199255".
the Y of the a80r0 is "1144440.2597765361".

[create a85r0]
a85r0 is a area. "An area (85) in r0".
the printed name of the a85r0 is "Room 0".
Understand "Area 85 in Room 0" as a85r0.
the parent of the a85r0 is r0.
the X of the a85r0 is "546941.9556509028".
the Y of the a85r0 is "1145135.8208926113".

[create a87r0]
a87r0 is a area. "An area (87) in r0".
the printed name of the a87r0 is "Room 0".
Understand "Area 87 in Room 0" as a87r0.
the parent of the a87r0 is r0.
the X of the a87r0 is "548572.516137043".
the Y of the a87r0 is "1145286.2231603272".

[create a89r0]
a89r0 is a area. "An area (89) in r0".
the printed name of the a89r0 is "Room 0".
Understand "Area 89 in Room 0" as a89r0.
the parent of the a89r0 is r0.
the X of the a89r0 is "545527.1865194305".
the Y of the a89r0 is "1144570.3208574706".

[create a90r0]
a90r0 is a area. "An area (90) in r0".
the printed name of the a90r0 is "Room 0".
Understand "Area 90 in Room 0" as a90r0.
the parent of the a90r0 is r0.
the X of the a90r0 is "549987.5".
the Y of the a90r0 is "1144568.1939858492".

[create a91r0]
a91r0 is a area. "An area (91) in r0".
the printed name of the a91r0 is "Room 0".
Understand "Area 91 in Room 0" as a91r0.
the parent of the a91r0 is r0.
the X of the a91r0 is "548526.5992826483".
the Y of the a91r0 is "1144604.1752513673".

[create a92r0]
a92r0 is a area. "An area (92) in r0".
the printed name of the a92r0 is "Room 0".
Understand "Area 92 in Room 0" as a92r0.
the parent of the a92r0 is r0.
the X of the a92r0 is "548504.3023199522".
the Y of the a92r0 is "1145140.5824958794".

[create a12r1]
a12r1 is a area. "An area (12) in r1".
the printed name of the a12r1 is "Room 1".
Understand "Area 12 in Room 1" as a12r1.
the parent of the a12r1 is r1.
the X of the a12r1 is "545120.0".
the Y of the a12r1 is "1146514.0".

[create a13r1]
a13r1 is a area. "An area (13) in r1".
the printed name of the a13r1 is "Room 1".
Understand "Area 13 in Room 1" as a13r1.
the parent of the a13r1 is r1.
the X of the a13r1 is "545150.9153780071".
the Y of the a13r1 is "1146488.1834621993".

[create a32r1]
a32r1 is a area. "An area (32) in r1".
the printed name of the a32r1 is "Room 1".
Understand "Area 32 in Room 1" as a32r1.
the parent of the a32r1 is r1.
the X of the a32r1 is "545111.0".
the Y of the a32r1 is "1146424.0".

[create a33r1]
a33r1 is a area. "An area (33) in r1".
the printed name of the a33r1 is "Room 1".
Understand "Area 33 in Room 1" as a33r1.
the parent of the a33r1 is r1.
the X of the a33r1 is "545153.06".
the Y of the a33r1 is "1146442.5189999999".

[create a61r1]
a61r1 is a area. "An area (61) in r1".
the printed name of the a61r1 is "Room 1".
Understand "Area 61 in Room 1" as a61r1.
the parent of the a61r1 is r1.
the X of the a61r1 is "546695.8043478262".
the Y of the a61r1 is "1146375.0".

[create a62r1]
a62r1 is a area. "An area (62) in r1".
the printed name of the a62r1 is "Room 1".
Understand "Area 62 in Room 1" as a62r1.
the parent of the a62r1 is r1.
the X of the a62r1 is "546659.1605611099".
the Y of the a62r1 is "1146383.567092667".

[create a65r1]
a65r1 is a area. "An area (65) in r1".
the printed name of the a65r1 is "Room 1".
Understand "Area 65 in Room 1" as a65r1.
the parent of the a65r1 is r1.
the X of the a65r1 is "545168.092671901".
the Y of the a65r1 is "1146459.8065726862".

[create a83r1]
a83r1 is a area. "An area (83) in r1".
the printed name of the a83r1 is "Room 1".
Understand "Area 83 in Room 1" as a83r1.
the parent of the a83r1 is r1.
the X of the a83r1 is "545527.6980415579".
the Y of the a83r1 is "1146333.2713725315".

[create a10r2]
a10r2 is a area. "An area (10) in r2".
the printed name of the a10r2 is "Room 2".
Understand "Area 10 in Room 2" as a10r2.
the parent of the a10r2 is r2.
the X of the a10r2 is "550402.0".
the Y of the a10r2 is "1146424.0".

[create a11r2]
a11r2 is a area. "An area (11) in r2".
the printed name of the a11r2 is "Room 2".
Understand "Area 11 in Room 2" as a11r2.
the parent of the a11r2 is r2.
the X of the a11r2 is "550359.94".
the Y of the a11r2 is "1146442.5189999999".

[create a14r2]
a14r2 is a area. "An area (14) in r2".
the printed name of the a14r2 is "Room 2".
Understand "Area 14 in Room 2" as a14r2.
the parent of the a14r2 is r2.
the X of the a14r2 is "548977.0".
the Y of the a14r2 is "1146514.0".

[create a15r2]
a15r2 is a area. "An area (15) in r2".
the printed name of the a15r2 is "Room 2".
Understand "Area 15 in Room 2" as a15r2.
the parent of the a15r2 is r2.
the X of the a15r2 is "548945.9123931624".
the Y of the a15r2 is "1146488.166239316".

[create a16r2]
a16r2 is a area. "An area (16) in r2".
the printed name of the a16r2 is "Room 2".
Understand "Area 16 in Room 2" as a16r2.
the parent of the a16r2 is r2.
the X of the a16r2 is "549582.0".
the Y of the a16r2 is "1146514.0".

[create a17r2]
a17r2 is a area. "An area (17) in r2".
the printed name of the a17r2 is "Room 2".
Understand "Area 17 in Room 2" as a17r2.
the parent of the a17r2 is r2.
the X of the a17r2 is "549613.0876068377".
the Y of the a17r2 is "1146488.1662393163".

[create a18r2]
a18r2 is a area. "An area (18) in r2".
the printed name of the a18r2 is "Room 2".
Understand "Area 18 in Room 2" as a18r2.
the parent of the a18r2 is r2.
the X of the a18r2 is "550393.0".
the Y of the a18r2 is "1146514.0".

[create a19r2]
a19r2 is a area. "An area (19) in r2".
the printed name of the a19r2 is "Room 2".
Understand "Area 19 in Room 2" as a19r2.
the parent of the a19r2 is r2.
the X of the a19r2 is "550361.9123931625".
the Y of the a19r2 is "1146488.1662393163".

[create a20r2]
a20r2 is a area. "An area (20) in r2".
the printed name of the a20r2 is "Room 2".
Understand "Area 20 in Room 2" as a20r2.
the parent of the a20r2 is r2.
the X of the a20r2 is "548166.0".
the Y of the a20r2 is "1146514.0".

[create a21r2]
a21r2 is a area. "An area (21) in r2".
the printed name of the a21r2 is "Room 2".
Understand "Area 21 in Room 2" as a21r2.
the parent of the a21r2 is r2.
the X of the a21r2 is "548197.8602595036".
the Y of the a21r2 is "1146506.4196750864".

[create a26r2]
a26r2 is a area. "An area (26) in r2".
the printed name of the a26r2 is "Room 2".
Understand "Area 26 in Room 2" as a26r2.
the parent of the a26r2 is r2.
the X of the a26r2 is "546670.0".
the Y of the a26r2 is "1145727.0".

[create a27r2]
a27r2 is a area. "An area (27) in r2".
the printed name of the a27r2 is "Room 2".
Understand "Area 27 in Room 2" as a27r2.
the parent of the a27r2 is r2.
the X of the a27r2 is "546817.6245056847".
the Y of the a27r2 is "1145689.375".

[create a28r2]
a28r2 is a area. "An area (28) in r2".
the printed name of the a28r2 is "Room 2".
Understand "Area 28 in Room 2" as a28r2.
the parent of the a28r2 is r2.
the X of the a28r2 is "547212.0".
the Y of the a28r2 is "1145730.0".

[create a29r2]
a29r2 is a area. "An area (29) in r2".
the printed name of the a29r2 is "Room 2".
Understand "Area 29 in Room 2" as a29r2.
the parent of the a29r2 is r2.
the X of the a29r2 is "547044.0931372547".
the Y of the a29r2 is "1145695.1666666665".

[create a34r2]
a34r2 is a area. "An area (34) in r2".
the printed name of the a34r2 is "Room 2".
Understand "Area 34 in Room 2" as a34r2.
the parent of the a34r2 is r2.
the X of the a34r2 is "548986.0".
the Y of the a34r2 is "1146424.0".

[create a35r2]
a35r2 is a area. "An area (35) in r2".
the printed name of the a35r2 is "Room 2".
Understand "Area 35 in Room 2" as a35r2.
the parent of the a35r2 is r2.
the X of the a35r2 is "548943.94".
the Y of the a35r2 is "1146442.5189999999".

[create a36r2]
a36r2 is a area. "An area (36) in r2".
the printed name of the a36r2 is "Room 2".
Understand "Area 36 in Room 2" as a36r2.
the parent of the a36r2 is r2.
the X of the a36r2 is "549573.0".
the Y of the a36r2 is "1146424.0".

[create a37r2]
a37r2 is a area. "An area (37) in r2".
the printed name of the a37r2 is "Room 2".
Understand "Area 37 in Room 2" as a37r2.
the parent of the a37r2 is r2.
the X of the a37r2 is "549615.0599999999".
the Y of the a37r2 is "1146442.5189999999".

[create a56r2]
a56r2 is a area. "An area (56) in r2".
the printed name of the a56r2 is "Room 2".
Understand "Area 56 in Room 2" as a56r2.
the parent of the a56r2 is r2.
the X of the a56r2 is "546889.0".
the Y of the a56r2 is "1146275.677419355".

[create a57r2]
a57r2 is a area. "An area (57) in r2".
the printed name of the a57r2 is "Room 2".
Understand "Area 57 in Room 2" as a57r2.
the parent of the a57r2 is r2.
the X of the a57r2 is "546907.9849724711".
the Y of the a57r2 is "1146244.5141828626".

[create a63r2]
a63r2 is a area. "An area (63) in r2".
the printed name of the a63r2 is "Room 2".
Understand "Area 63 in Room 2" as a63r2.
the parent of the a63r2 is r2.
the X of the a63r2 is "549630.1393831554".
the Y of the a63r2 is "1146459.8602906286".

[create a64r2]
a64r2 is a area. "An area (64) in r2".
the printed name of the a64r2 is "Room 2".
Understand "Area 64 in Room 2" as a64r2.
the parent of the a64r2 is r2.
the X of the a64r2 is "550344.8606168446".
the Y of the a64r2 is "1146459.8602906289".

[create a66r2]
a66r2 is a area. "An area (66) in r2".
the printed name of the a66r2 is "Room 2".
Understand "Area 66 in Room 2" as a66r2.
the parent of the a66r2 is r2.
the X of the a66r2 is "548928.8606168446".
the Y of the a66r2 is "1146459.8602906289".

[create a67r2]
a67r2 is a area. "An area (67) in r2".
the printed name of the a67r2 is "Room 2".
Understand "Area 67 in Room 2" as a67r2.
the parent of the a67r2 is r2.
the X of the a67r2 is "548202.5737633365".
the Y of the a67r2 is "1146449.1688060723".

[create a70r2]
a70r2 is a area. "An area (70) in r2".
the printed name of the a70r2 is "Room 2".
Understand "Area 70 in Room 2" as a70r2.
the parent of the a70r2 is r2.
the X of the a70r2 is "546940.921357456".
the Y of the a70r2 is "1145661.0075997405".

[create a71r2]
a71r2 is a area. "An area (71) in r2".
the printed name of the a71r2 is "Room 2".
Understand "Area 71 in Room 2" as a71r2.
the parent of the a71r2 is r2.
the X of the a71r2 is "546944.9953661207".
the Y of the a71r2 is "1145677.339554399".

[create a81r2]
a81r2 is a area. "An area (81) in r2".
the printed name of the a81r2 is "Room 2".
Understand "Area 81 in Room 2" as a81r2.
the parent of the a81r2 is r2.
the X of the a81r2 is "548556.2760341757".
the Y of the a81r2 is "1146325.2598443679".

[create a82r2]
a82r2 is a area. "An area (82) in r2".
the printed name of the a82r2 is "Room 2".
Understand "Area 82 in Room 2" as a82r2.
the parent of the a82r2 is r2.
the X of the a82r2 is "547014.5495530016".
the Y of the a82r2 is "1146126.3090731262".

[create a84r2]
a84r2 is a area. "An area (84) in r2".
the printed name of the a84r2 is "Room 2".
Understand "Area 84 in Room 2" as a84r2.
the parent of the a84r2 is r2.
the X of the a84r2 is "549987.5".
the Y of the a84r2 is "1146335.7211084906".

[create a86r2]
a86r2 is a area. "An area (86) in r2".
the printed name of the a86r2 is "Room 2".
Understand "Area 86 in Room 2" as a86r2.
the parent of the a86r2 is r2.
the X of the a86r2 is "548584.832709364".
the Y of the a86r2 is "1145494.0".

[create a88r2]
a88r2 is a area. "An area (88) in r2".
the printed name of the a88r2 is "Room 2".
Understand "Area 88 in Room 2" as a88r2.
the parent of the a88r2 is r2.
the X of the a88r2 is "549974.4643321299".
the Y of the a88r2 is "1145494.0".

[create a30r3]
a30r3 is a area. "An area (30) in r3".
the printed name of the a30r3 is "Room 3".
Understand "Area 30 in Room 3" as a30r3.
the parent of the a30r3 is r3.
the X of the a30r3 is "546889.0".
the Y of the a30r3 is "1146375.0".

[create a31r3]
a31r3 is a area. "An area (31) in r3".
the printed name of the a31r3 is "Room 3".
Understand "Area 31 in Room 3" as a31r3.
the parent of the a31r3 is r3.
the X of the a31r3 is "546856.8007246377".
the Y of the a31r3 is "1146350.1693548388".

[create a56r3]
a56r3 is a area. "An area (56) in r3".
the printed name of the a56r3 is "Room 3".
Understand "Area 56 in Room 3" as a56r3.
the parent of the a56r3 is r3.
the X of the a56r3 is "546889.0".
the Y of the a56r3 is "1146275.677419355".

[create a58r3]
a58r3 is a area. "An area (58) in r3".
the printed name of the a58r3 is "Room 3".
Understand "Area 58 in Room 3" as a58r3.
the parent of the a58r3 is r3.
the X of the a58r3 is "546873.7064157481".
the Y of the a58r3 is "1146300.7813579345".

[create a59r3]
a59r3 is a area. "An area (59) in r3".
the printed name of the a59r3 is "Room 3".
Understand "Area 59 in Room 3" as a59r3.
the parent of the a59r3 is r3.
the X of the a59r3 is "546714.0".
the Y of the a59r3 is "1146366.0".

[create a60r3]
a60r3 is a area. "An area (60) in r3".
the printed name of the a60r3 is "Room 3".
Understand "Area 60 in Room 3" as a60r3.
the parent of the a60r3 is r3.
the X of the a60r3 is "546792.4021739131".
the Y of the a60r3 is "1146363.5525401558".

[create a61r3]
a61r3 is a area. "An area (61) in r3".
the printed name of the a61r3 is "Room 3".
Understand "Area 61 in Room 3" as a61r3.
the parent of the a61r3 is r3.
the X of the a61r3 is "546695.8043478262".
the Y of the a61r3 is "1146375.0".

[create a72r3]
a72r3 is a area. "An area (72) in r3".
the printed name of the a72r3 is "Room 3".
Understand "Area 72 in Room 3" as a72r3.
the parent of the a72r3 is r3.
the X of the a72r3 is "546850.8286071355".
the Y of the a72r3 is "1146342.4249786977".

[create d0]
d0 is a door. "A door between a89r0 and a83r1".
d0 is north of a89r0 and south of a83r1.

[create d2]
d2 is a door. "A door between a85r0 and a70r2".
d2 is north of a85r0 and south of a70r2.

[create d3]
d3 is a door. "A door between a87r0 and a86r2".
d3 is north of a87r0 and south of a86r2.

[create d4]
d4 is a door. "A door between a90r0 and a88r2".
d4 is north of a90r0 and south of a88r2.

northeast of a0r0 is southwest of a1r0.

northeast of a2r0 is southwest of a3r0.

southeast of a4r0 is northwest of a5r0.

southeast of a6r0 is northwest of a7r0.

northwest of a8r0 is southeast of a9r0.

northwest of a10r2 is southeast of a11r2.

southeast of a12r1 is northwest of a13r1.

southwest of a14r2 is northeast of a15r2.

southeast of a16r2 is northwest of a17r2.

southwest of a18r2 is northeast of a19r2.

east of a20r2 is west of a21r2.

southeast of a22r0 is northwest of a23r0.

southwest of a24r0 is northeast of a25r0.

east of a26r2 is west of a27r2.

west of a28r2 is east of a29r2.

southwest of a30r3 is northeast of a31r3.

northeast of a32r1 is southwest of a33r1.

northwest of a34r2 is southeast of a35r2.

northeast of a36r2 is southwest of a37r2.

northwest of a38r0 is southeast of a39r0.

southwest of a40r0 is northeast of a41r0.

southwest of a42r0 is northeast of a43r0.

north of a44r0 is south of a45r0.

northwest of a46r0 is southeast of a47r0.

north of a48r0 is south of a49r0.

northeast of a50r0 is southwest of a51r0.

northeast of a52r0 is southwest of a53r0.

west of a54r0 is east of a55r0.

southeast of a56r2 is northwest of a57r2.

east of a56r2 is west of a56r2.

east of a59r3 is west of a60r3.

west of a61r1 is east of a62r1.

northeast of a37r2 is southwest of a63r2.

northwest of a11r2 is southeast of a64r2.

southeast of a13r1 is northwest of a65r1.

southwest of a15r2 is northeast of a66r2.

southeast of a17r2 is northwest of a63r2.

southwest of a19r2 is northeast of a64r2.

south of a21r2 is north of a67r2.

southeast of a23r0 is northwest of a68r0.

southwest of a25r0 is northeast of a69r0.

east of a27r2 is west of a70r2.

north of a70r2 is south of a71r2.

west of a29r2 is east of a71r2.

southwest of a31r3 is northeast of a72r3.

northeast of a33r1 is southwest of a65r1.

northwest of a35r2 is southeast of a66r2.

southwest of a41r0 is northeast of a73r0.

southwest of a43r0 is northeast of a74r0.

north of a45r0 is south of a75r0.

northwest of a47r0 is southeast of a76r0.

northeast of a49r0 is southwest of a77r0.

south of a77r0 is north of a78r0.

northeast of a51r0 is southwest of a68r0.

northwest of a53r0 is southeast of a69r0.

northeast of a1r0 is southwest of a79r0.

northeast of a3r0 is southwest of a80r0.

southeast of a5r0 is northwest of a79r0.

southeast of a7r0 is northwest of a80r0.

northwest of a9r0 is southeast of a73r0.

northwest of a39r0 is southeast of a74r0.

east of a67r2 is west of a81r2.

west of a67r2 is east of a82r2.

northwest of a58r3 is southeast of a72r3.

east of a60r3 is west of a72r3.

west of a62r1 is east of a83r1.

east of a63r2 is west of a84r2.

west of a64r2 is east of a84r2.

east of a65r1 is west of a83r1.

west of a66r2 is east of a81r2.

west of a55r0 is east of a85r0.

east of a86r2 is west of a88r2.

south of a84r2 is north of a88r2.

south of a81r2 is north of a86r2.

north of a71r2 is south of a82r2.

southeast of a57r2 is northwest of a82r2.

northeast of a76r0 is southwest of a78r0.

north of a91r0 is south of a92r0.

north of a77r0 is south of a85r0.

east of a68r0 is west of a92r0.

west of a69r0 is east of a87r0.

southwest of a87r0 is northeast of a92r0.

west of a74r0 is east of a90r0.

west of a73r0 is east of a91r0.

east of a80r0 is west of a90r0.

east of a79r0 is west of a89r0.

east of a75r0 is west of a91r0.

west of a76r0 is east of a89r0.

northwest of a75r0 is southeast of a78r0.

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
the player is in a13r1.

the orientation of the player is 6.
