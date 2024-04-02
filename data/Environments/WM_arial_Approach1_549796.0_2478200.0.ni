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

[create r8]
r8 is a indoor_room. "Room 8".

[create a2r0]
a2r0 is a area. "An area (2) in r0".
the printed name of the a2r0 is "Room 0".
Understand "Area 2 in Room 0" as a2r0.
the parent of the a2r0 is r0.
the X of the a2r0 is "545223.1371343824".
the Y of the a2r0 is "2480298.0705300593".

[create a28r0]
a28r0 is a area. "An area (28) in r0".
the printed name of the a28r0 is "Room 0".
Understand "Area 28 in Room 0" as a28r0.
the parent of the a28r0 is r0.
the X of the a28r0 is "545075.2500000002".
the Y of the a28r0 is "2480409.832708333".

[create a31r0]
a31r0 is a area. "An area (31) in r0".
the printed name of the a31r0 is "Room 0".
Understand "Area 31 in Room 0" as a31r0.
the parent of the a31r0 is r0.
the X of the a31r0 is "545298.7500000003".
the Y of the a31r0 is "2480392.8165930705".

[create a41r0]
a41r0 is a area. "An area (41) in r0".
the printed name of the a41r0 is "Room 0".
Understand "Area 41 in Room 0" as a41r0.
the parent of the a41r0 is r0.
the X of the a41r0 is "545336.0".
the Y of the a41r0 is "2480440.0".

[create a44r0]
a44r0 is a area. "An area (44) in r0".
the printed name of the a44r0 is "Room 0".
Understand "Area 44 in Room 0" as a44r0.
the parent of the a44r0 is r0.
the X of the a44r0 is "545038.0".
the Y of the a44r0 is "2480440.0".

[create a3r1]
a3r1 is a area. "An area (3) in r1".
the printed name of the a3r1 is "Room 1".
Understand "Area 3 in Room 1" as a3r1.
the parent of the a3r1 is r1.
the X of the a3r1 is "545738.6462401489".
the Y of the a3r1 is "2478353.4981471454".

[create a6r1]
a6r1 is a area. "An area (6) in r1".
the printed name of the a6r1 is "Room 1".
Understand "Area 6 in Room 1" as a6r1.
the parent of the a6r1 is r1.
the X of the a6r1 is "545800.9412801259".
the Y of the a6r1 is "2478354.6105585736".

[create a18r1]
a18r1 is a area. "An area (18) in r1".
the printed name of the a18r1 is "Room 1".
Understand "Area 18 in Room 1" as a18r1.
the parent of the a18r1 is r1.
the X of the a18r1 is "545658.125".
the Y of the a18r1 is "2478249.5247135414".

[create a19r1]
a19r1 is a area. "An area (19) in r1".
the printed name of the a19r1 is "Room 1".
Understand "Area 19 in Room 1" as a19r1.
the parent of the a19r1 is r1.
the X of the a19r1 is "545880.875".
the Y of the a19r1 is "2478250.099474627".

[create a50r1]
a50r1 is a area. "An area (50) in r1".
the printed name of the a50r1 is "Room 1".
Understand "Area 50 in Room 1" as a50r1.
the parent of the a50r1 is r1.
the X of the a50r1 is "545621.0".
the Y of the a50r1 is "2478200.0".

[create a51r1]
a51r1 is a area. "An area (51) in r1".
the printed name of the a51r1 is "Room 1".
Understand "Area 51 in Room 1" as a51r1.
the parent of the a51r1 is r1.
the X of the a51r1 is "545918.0".
the Y of the a51r1 is "2478200.0".

[create a7r2]
a7r2 is a area. "An area (7) in r2".
the printed name of the a7r2 is "Room 2".
Understand "Area 7 in Room 2" as a7r2.
the parent of the a7r2 is r2.
the X of the a7r2 is "546387.5565103934".
the Y of the a7r2 is "2480302.537885611".

[create a21r2]
a21r2 is a area. "An area (21) in r2".
the printed name of the a21r2 is "Room 2".
Understand "Area 21 in Room 2" as a21r2.
the parent of the a21r2 is r2.
the X of the a21r2 is "546523.8305982086".
the Y of the a21r2 is "2480298.8947038352".

[create a29r2]
a29r2 is a area. "An area (29) in r2".
the printed name of the a29r2 is "Room 2".
Understand "Area 29 in Room 2" as a29r2.
the parent of the a29r2 is r2.
the X of the a29r2 is "546596.9".
the Y of the a29r2 is "2480389.466754617".

[create a30r2]
a30r2 is a area. "An area (30) in r2".
the printed name of the a30r2 is "Room 2".
Understand "Area 30 in Room 2" as a30r2.
the parent of the a30r2 is r2.
the X of the a30r2 is "546316.1000000001".
the Y of the a30r2 is "2480389.5239782464".

[create a39r2]
a39r2 is a area. "An area (39) in r2".
the printed name of the a39r2 is "Room 2".
Understand "Area 39 in Room 2" as a39r2.
the parent of the a39r2 is r2.
the X of the a39r2 is "546632.0".
the Y of the a39r2 is "2480440.0".

[create a40r2]
a40r2 is a area. "An area (40) in r2".
the printed name of the a40r2 is "Room 2".
Understand "Area 40 in Room 2" as a40r2.
the parent of the a40r2 is r2.
the X of the a40r2 is "546281.0".
the Y of the a40r2 is "2480440.0".

[create a0r3]
a0r3 is a area. "An area (0) in r3".
the printed name of the a0r3 is "Room 3".
Understand "Area 0 in Room 3" as a0r3.
the parent of the a0r3 is r3.
the X of the a0r3 is "547145.596495711".
the Y of the a0r3 is "2478354.3324220222".

[create a15r3]
a15r3 is a area. "An area (15) in r3".
the printed name of the a15r3 is "Room 3".
Understand "Area 15 in Room 3" as a15r3.
the parent of the a15r3 is r3.
the X of the a15r3 is "547017.0".
the Y of the a15r3 is "2478249.422623434".

[create a16r3]
a16r3 is a area. "An area (16) in r3".
the printed name of the a16r3 is "Room 3".
Understand "Area 16 in Room 3" as a16r3.
the parent of the a16r3 is r3.
the X of the a16r3 is "547097.2576810801".
the Y of the a16r3 is "2478353.704645209".

[create a17r3]
a17r3 is a area. "An area (17) in r3".
the printed name of the a17r3 is "Room 3".
Understand "Area 17 in Room 3" as a17r3.
the parent of the a17r3 is r3.
the X of the a17r3 is "547227.0".
the Y of the a17r3 is "2478248.8087123735".

[create a48r3]
a48r3 is a area. "An area (48) in r3".
the printed name of the a48r3 is "Room 3".
Understand "Area 48 in Room 3" as a48r3.
the parent of the a48r3 is r3.
the X of the a48r3 is "546982.0".
the Y of the a48r3 is "2478200.0".

[create a49r3]
a49r3 is a area. "An area (49) in r3".
the printed name of the a49r3 is "Room 3".
Understand "Area 49 in Room 3" as a49r3.
the parent of the a49r3 is r3.
the X of the a49r3 is "547262.0".
the Y of the a49r3 is "2478200.0".

[create a1r4]
a1r4 is a area. "An area (1) in r4".
the printed name of the a1r4 is "Room 4".
Understand "Area 1 in Room 4" as a1r4.
the parent of the a1r4 is r4.
the X of the a1r4 is "547733.4302703282".
the Y of the a1r4 is "2480058.788369159".

[create a8r4]
a8r4 is a area. "An area (8) in r4".
the printed name of the a8r4 is "Room 4".
Understand "Area 8 in Room 4" as a8r4.
the parent of the a8r4 is r4.
the X of the a8r4 is "548017.3606929516".
the Y of the a8r4 is "2480274.591941147".

[create a10r4]
a10r4 is a area. "An area (10) in r4".
the printed name of the a10r4 is "Room 4".
Understand "Area 10 in Room 4" as a10r4.
the parent of the a10r4 is r4.
the X of the a10r4 is "547758.1754127733".
the Y of the a10r4 is "2480202.652806954".

[create a11r4]
a11r4 is a area. "An area (11) in r4".
the printed name of the a11r4 is "Room 4".
Understand "Area 11 in Room 4" as a11r4.
the parent of the a11r4 is r4.
the X of the a11r4 is "548101.2500000005".
the Y of the a11r4 is "2480388.924728417".

[create a12r4]
a12r4 is a area. "An area (12) in r4".
the printed name of the a12r4 is "Room 4".
Understand "Area 12 in Room 4" as a12r4.
the parent of the a12r4 is r4.
the X of the a12r4 is "547603.9999999997".
the Y of the a12r4 is "2480392.444031076".

[create a37r4]
a37r4 is a area. "An area (37) in r4".
the printed name of the a37r4 is "Room 4".
Understand "Area 37 in Room 4" as a37r4.
the parent of the a37r4 is r4.
the X of the a37r4 is "548135.0".
the Y of the a37r4 is "2480440.0".

[create a38r4]
a38r4 is a area. "An area (38) in r4".
the printed name of the a38r4 is "Room 4".
Understand "Area 38 in Room 4" as a38r4.
the parent of the a38r4 is r4.
the X of the a38r4 is "547573.0".
the Y of the a38r4 is "2480440.0".

[create a9r5]
a9r5 is a area. "An area (9) in r5".
the printed name of the a9r5 is "Room 5".
Understand "Area 9 in Room 5" as a9r5.
the parent of the a9r5 is r5.
the X of the a9r5 is "548725.624643379".
the Y of the a9r5 is "2478359.8817214696".

[create a13r5]
a13r5 is a area. "An area (13) in r5".
the printed name of the a13r5 is "Room 5".
Understand "Area 13 in Room 5" as a13r5.
the parent of the a13r5 is r5.
the X of the a13r5 is "548766.1926899109".
the Y of the a13r5 is "2478357.834122907".

[create a14r5]
a14r5 is a area. "An area (14) in r5".
the printed name of the a14r5 is "Room 5".
Understand "Area 14 in Room 5" as a14r5.
the parent of the a14r5 is r5.
the X of the a14r5 is "548843.25".
the Y of the a14r5 is "2478251.484325186".

[create a27r5]
a27r5 is a area. "An area (27) in r5".
the printed name of the a27r5 is "Room 5".
Understand "Area 27 in Room 5" as a27r5.
the parent of the a27r5 is r5.
the X of the a27r5 is "548646.75".
the Y of the a27r5 is "2478251.7428626004".

[create a46r5]
a46r5 is a area. "An area (46) in r5".
the printed name of the a46r5 is "Room 5".
Understand "Area 46 in Room 5" as a46r5.
the parent of the a46r5 is r5.
the X of the a46r5 is "548614.0".
the Y of the a46r5 is "2478200.0".

[create a47r5]
a47r5 is a area. "An area (47) in r5".
the printed name of the a47r5 is "Room 5".
Understand "Area 47 in Room 5" as a47r5.
the parent of the a47r5 is r5.
the X of the a47r5 is "548876.0".
the Y of the a47r5 is "2478200.0".

[create a5r6]
a5r6 is a area. "An area (5) in r6".
the printed name of the a5r6 is "Room 6".
Understand "Area 5 in Room 6" as a5r6.
the parent of the a5r6 is r6.
the X of the a5r6 is "549607.0934343645".
the Y of the a5r6 is "2480250.5944565693".

[create a20r6]
a20r6 is a area. "An area (20) in r6".
the printed name of the a20r6 is "Room 6".
Understand "Area 20 in Room 6" as a20r6.
the parent of the a20r6 is r6.
the X of the a20r6 is "549532.0511527922".
the Y of the a20r6 is "2480261.748260341".

[create a32r6]
a32r6 is a area. "An area (32) in r6".
the printed name of the a32r6 is "Room 6".
Understand "Area 32 in Room 6" as a32r6.
the parent of the a32r6 is r6.
the X of the a32r6 is "549443.2000000004".
the Y of the a32r6 is "2480385.378981202".

[create a33r6]
a33r6 is a area. "An area (33) in r6".
the printed name of the a33r6 is "Room 6".
Understand "Area 33 in Room 6" as a33r6.
the parent of the a33r6 is r6.
the X of the a33r6 is "549756.8".
the Y of the a33r6 is "2480401.379310345".

[create a42r6]
a42r6 is a area. "An area (42) in r6".
the printed name of the a42r6 is "Room 6".
Understand "Area 42 in Room 6" as a42r6.
the parent of the a42r6 is r6.
the X of the a42r6 is "549404.0".
the Y of the a42r6 is "2480440.0".

[create a43r6]
a43r6 is a area. "An area (43) in r6".
the printed name of the a43r6 is "Room 6".
Understand "Area 43 in Room 6" as a43r6.
the parent of the a43r6 is r6.
the X of the a43r6 is "549796.0".
the Y of the a43r6 is "2480440.0".

[create a4r7]
a4r7 is a area. "An area (4) in r7".
the printed name of the a4r7 is "Room 7".
Understand "Area 4 in Room 7" as a4r7.
the parent of the a4r7 is r7.
the X of the a4r7 is "549655.3794133377".
the Y of the a4r7 is "2478341.242485417".

[create a23r7]
a23r7 is a area. "An area (23) in r7".
the printed name of the a23r7 is "Room 7".
Understand "Area 23 in Room 7" as a23r7.
the parent of the a23r7 is r7.
the X of the a23r7 is "549550.125".
the Y of the a23r7 is "2478239.208333333".

[create a24r7]
a24r7 is a area. "An area (24) in r7".
the printed name of the a24r7 is "Room 7".
Understand "Area 24 in Room 7" as a24r7.
the parent of the a24r7 is r7.
the X of the a24r7 is "549760.875".
the Y of the a24r7 is "2478238.620689655".

[create a34r7]
a34r7 is a area. "An area (34) in r7".
the printed name of the a34r7 is "Room 7".
Understand "Area 34 in Room 7" as a34r7.
the parent of the a34r7 is r7.
the X of the a34r7 is "549515.0".
the Y of the a34r7 is "2478200.0".

[create a35r7]
a35r7 is a area. "An area (35) in r7".
the printed name of the a35r7 is "Room 7".
Understand "Area 35 in Room 7" as a35r7.
the parent of the a35r7 is r7.
the X of the a35r7 is "549796.0".
the Y of the a35r7 is "2478200.0".

[create a22r8]
a22r8 is a area. "An area (22) in r8".
the printed name of the a22r8 is "Room 8".
Understand "Area 22 in Room 8" as a22r8.
the parent of the a22r8 is r8.
the X of the a22r8 is "547837.4621130738".
the Y of the a22r8 is "2478341.0485422974".

[create a25r8]
a25r8 is a area. "An area (25) in r8".
the printed name of the a25r8 is "Room 8".
Understand "Area 25 in Room 8" as a25r8.
the parent of the a25r8 is r8.
the X of the a25r8 is "547732.125".
the Y of the a25r8 is "2478238.4742951905".

[create a26r8]
a26r8 is a area. "An area (26) in r8".
the printed name of the a26r8 is "Room 8".
Understand "Area 26 in Room 8" as a26r8.
the parent of the a26r8 is r8.
the X of the a26r8 is "547942.875".
the Y of the a26r8 is "2478238.2600000002".

[create a36r8]
a36r8 is a area. "An area (36) in r8".
the printed name of the a36r8 is "Room 8".
Understand "Area 36 in Room 8" as a36r8.
the parent of the a36r8 is r8.
the X of the a36r8 is "547697.0".
the Y of the a36r8 is "2478200.0".

[create a45r8]
a45r8 is a area. "An area (45) in r8".
the printed name of the a45r8 is "Room 8".
Understand "Area 45 in Room 8" as a45r8.
the parent of the a45r8 is r8.
the X of the a45r8 is "547978.0".
the Y of the a45r8 is "2478200.0".

[create d0]
d0 is a door. "A door between a13r5 and a20r6".
d0 is north of a13r5 and south of a20r6.

[create d2]
d2 is a door. "A door between a21r2 and a16r3".
d2 is south of a21r2 and north of a16r3.

[create d4]
d4 is a door. "A door between a1r4 and a22r8".
d4 is south of a1r4 and north of a22r8.

[create d6]
d6 is a door. "A door between a0r3 and a1r4".
d6 is northeast of a0r3 and southwest of a1r4.

[create d8]
d8 is a door. "A door between a2r0 and a3r1".
d8 is south of a2r0 and north of a3r1.

[create d10]
d10 is a door. "A door between a5r6 and a4r7".
d10 is south of a5r6 and north of a4r7.

[create d12]
d12 is a door. "A door between a6r1 and a7r2".
d12 is north of a6r1 and south of a7r2.

[create d14]
d14 is a door. "A door between a8r4 and a9r5".
d14 is south of a8r4 and north of a9r5.

north of a1r4 is south of a10r4.

west of a8r4 is east of a10r4.

northeast of a8r4 is southwest of a11r4.

northwest of a10r4 is southeast of a12r4.

southeast of a13r5 is northwest of a14r5.

northeast of a15r3 is southwest of a16r3.

southeast of a0r3 is northwest of a17r3.

southwest of a3r1 is northeast of a18r1.

southeast of a6r1 is northwest of a19r1.

southwest of a4r7 is northeast of a23r7.

southeast of a4r7 is northwest of a24r7.

southwest of a22r8 is northeast of a25r8.

southeast of a22r8 is northwest of a26r8.

southwest of a9r5 is northeast of a27r5.

northwest of a2r0 is southeast of a28r0.

west of a5r6 is east of a20r6.

northeast of a21r2 is southwest of a29r2.

northwest of a7r2 is southeast of a30r2.

northeast of a2r0 is southwest of a31r0.

northwest of a20r6 is southeast of a32r6.

east of a7r2 is west of a21r2.

northeast of a5r6 is southwest of a33r6.

southwest of a23r7 is northeast of a34r7.

southeast of a24r7 is northwest of a35r7.

southwest of a25r8 is northeast of a36r8.

northeast of a11r4 is southwest of a37r4.

northwest of a12r4 is southeast of a38r4.

northeast of a29r2 is southwest of a39r2.

northwest of a30r2 is southeast of a40r2.

northeast of a31r0 is southwest of a41r0.

northwest of a32r6 is southeast of a42r6.

northeast of a33r6 is southwest of a43r6.

northwest of a28r0 is southeast of a44r0.

southeast of a26r8 is northwest of a45r8.

southwest of a27r5 is northeast of a46r5.

southeast of a14r5 is northwest of a47r5.

southwest of a15r3 is northeast of a48r3.

southeast of a17r3 is northwest of a49r3.

southwest of a18r1 is northeast of a50r1.

southeast of a19r1 is northwest of a51r1.

west of a0r3 is east of a16r3.

east of a3r1 is west of a6r1.

east of a9r5 is west of a13r5.

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
the player is in a35r7.

the orientation of the player is 6.
