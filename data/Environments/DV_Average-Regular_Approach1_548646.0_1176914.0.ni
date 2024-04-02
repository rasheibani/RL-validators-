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
the X of the a0r0 is "547887.0".
the Y of the a0r0 is "1176914.0".

[create a1r0]
a1r0 is a area. "An area (1) in r0".
the printed name of the a1r0 is "Room 0".
Understand "Area 1 in Room 0" as a1r0.
the parent of the a1r0 is r0.
the X of the a1r0 is "547918.0876068376".
the Y of the a1r0 is "1176888.1662393163".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "547878.0".
the Y of the a6r0 is "1176824.0".

[create a7r0]
a7r0 is a area. "An area (7) in r0".
the printed name of the a7r0 is "Room 0".
Understand "Area 7 in Room 0" as a7r0.
the parent of the a7r0 is r0.
the X of the a7r0 is "547918.7622699386".
the Y of the a7r0 is "1176842.648773006".

[create a8r0]
a8r0 is a area. "An area (8) in r0".
the printed name of the a8r0 is "Room 0".
Understand "Area 8 in Room 0" as a8r0.
the parent of the a8r0 is r0.
the X of the a8r0 is "548655.0".
the Y of the a8r0 is "1176824.0".

[create a9r0]
a9r0 is a area. "An area (9) in r0".
the printed name of the a9r0 is "Room 0".
Understand "Area 9 in Room 0" as a9r0.
the parent of the a9r0 is r0.
the X of the a9r0 is "548616.1323529412".
the Y of the a9r0 is "1176842.8382352942".

[create a16r0]
a16r0 is a area. "An area (16) in r0".
the printed name of the a16r0 is "Room 0".
Understand "Area 16 in Room 0" as a16r0.
the parent of the a16r0 is r0.
the X of the a16r0 is "548646.0".
the Y of the a16r0 is "1176914.0".

[create a17r0]
a17r0 is a area. "An area (17) in r0".
the printed name of the a17r0 is "Room 0".
Understand "Area 17 in Room 0" as a17r0.
the parent of the a17r0 is r0.
the X of the a17r0 is "548614.9123931625".
the Y of the a17r0 is "1176888.1662393163".

[create a30r0]
a30r0 is a area. "An area (30) in r0".
the printed name of the a30r0 is "Room 0".
Understand "Area 30 in Room 0" as a30r0.
the parent of the a30r0 is r0.
the X of the a30r0 is "548598.8307569794".
the Y of the a30r0 is "1176861.470723253".

[create a31r0]
a31r0 is a area. "An area (31) in r0".
the printed name of the a31r0 is "Room 0".
Understand "Area 31 in Room 0" as a31r0.
the parent of the a31r0 is r0.
the X of the a31r0 is "547934.5405048296".
the Y of the a31r0 is "1176860.8544286494".

[create a39r0]
a39r0 is a area. "An area (39) in r0".
the printed name of the a39r0 is "Room 0".
Understand "Area 39 in Room 0" as a39r0.
the parent of the a39r0 is r0.
the X of the a39r0 is "548274.4895726717".
the Y of the a39r0 is "1176772.4549042068".

[create a18r1]
a18r1 is a area. "An area (18) in r1".
the printed name of the a18r1 is "Room 1".
Understand "Area 18 in Room 1" as a18r1.
the parent of the a18r1 is r1.
the X of the a18r1 is "545098.0".
the Y of the a18r1 is "1174791.0".

[create a19r1]
a19r1 is a area. "An area (19) in r1".
the printed name of the a19r1 is "Room 1".
Understand "Area 19 in Room 1" as a19r1.
the parent of the a19r1 is r1.
the X of the a19r1 is "545129.2137161086".
the Y of the a19r1 is "1174816.633971292".

[create a20r1]
a20r1 is a area. "An area (20) in r1".
the printed name of the a20r1 is "Room 1".
Understand "Area 20 in Room 1" as a20r1.
the parent of the a20r1 is r1.
the X of the a20r1 is "545089.0".
the Y of the a20r1 is "1174880.0".

[create a21r1]
a21r1 is a area. "An area (21) in r1".
the printed name of the a21r1 is "Room 1".
Understand "Area 21 in Room 1" as a21r1.
the parent of the a21r1 is r1.
the X of the a21r1 is "545120.0059757696".
the Y of the a21r1 is "1174860.657907662".

[create a22r1]
a22r1 is a area. "An area (22) in r1".
the printed name of the a22r1 is "Room 1".
Understand "Area 22 in Room 1" as a22r1.
the parent of the a22r1 is r1.
the X of the a22r1 is "547736.0".
the Y of the a22r1 is "1174800.0".

[create a23r1]
a23r1 is a area. "An area (23) in r1".
the printed name of the a23r1 is "Room 1".
Understand "Area 23 in Room 1" as a23r1.
the parent of the a23r1 is r1.
the X of the a23r1 is "547693.4517163367".
the Y of the a23r1 is "1174849.4252993807".

[create a24r1]
a24r1 is a area. "An area (24) in r1".
the printed name of the a24r1 is "Room 1".
Understand "Area 24 in Room 1" as a24r1.
the parent of the a24r1 is r1.
the X of the a24r1 is "546604.0".
the Y of the a24r1 is "1174800.0".

[create a25r1]
a25r1 is a area. "An area (25) in r1".
the printed name of the a25r1 is "Room 1".
Understand "Area 25 in Room 1" as a25r1.
the parent of the a25r1 is r1.
the X of the a25r1 is "546567.875".
the Y of the a25r1 is "1174885.6511487633".

[create a28r1]
a28r1 is a area. "An area (28) in r1".
the printed name of the a28r1 is "Room 1".
Understand "Area 28 in Room 1" as a28r1.
the parent of the a28r1 is r1.
the X of the a28r1 is "547435.0".
the Y of the a28r1 is "1174772.0".

[create a29r1]
a29r1 is a area. "An area (29) in r1".
the printed name of the a29r1 is "Room 1".
Understand "Area 29 in Room 1" as a29r1.
the parent of the a29r1 is r1.
the X of the a29r1 is "547468.1199407121".
the Y of the a29r1 is "1174823.929387344".

[create a36r1]
a36r1 is a area. "An area (36) in r1".
the printed name of the a36r1 is "Room 1".
Understand "Area 36 in Room 1" as a36r1.
the parent of the a36r1 is r1.
the X of the a36r1 is "545143.389199255".
the Y of the a36r1 is "1174840.2597765361".

[create a37r1]
a37r1 is a area. "An area (37) in r1".
the printed name of the a37r1 is "Room 1".
Understand "Area 37 in Room 1" as a37r1.
the parent of the a37r1 is r1.
the X of the a37r1 is "547629.9629832634".
the Y of the a37r1 is "1174928.2779362001".

[create a38r1]
a38r1 is a area. "An area (38) in r1".
the printed name of the a38r1 is "Room 1".
Understand "Area 38 in Room 1" as a38r1.
the parent of the a38r1 is r1.
the X of the a38r1 is "546546.4065988241".
the Y of the a38r1 is "1174936.1959030344".

[create a40r1]
a40r1 is a area. "An area (40) in r1".
the printed name of the a40r1 is "Room 1".
Understand "Area 40 in Room 1" as a40r1.
the parent of the a40r1 is r1.
the X of the a40r1 is "545527.1865194305".
the Y of the a40r1 is "1174970.3208574704".

[create a43r1]
a43r1 is a area. "An area (43) in r1".
the printed name of the a43r1 is "Room 1".
Understand "Area 43 in Room 1" as a43r1.
the parent of the a43r1 is r1.
the X of the a43r1 is "547145.6287514417".
the Y of the a43r1 is "1175675.1570684114".

[create a44r1]
a44r1 is a area. "An area (44) in r1".
the printed name of the a44r1 is "Room 1".
Understand "Area 44 in Room 1" as a44r1.
the parent of the a44r1 is r1.
the X of the a44r1 is "547539.0369901187".
the Y of the a44r1 is "1174933.4740562679".

[create a2r2]
a2r2 is a area. "An area (2) in r2".
the printed name of the a2r2 is "Room 2".
Understand "Area 2 in Room 2" as a2r2.
the parent of the a2r2 is r2.
the X of the a2r2 is "545111.0".
the Y of the a2r2 is "1176824.0".

[create a3r2]
a3r2 is a area. "An area (3) in r2".
the printed name of the a3r2 is "Room 2".
Understand "Area 3 in Room 2" as a3r2.
the parent of the a3r2 is r2.
the X of the a3r2 is "545153.06".
the Y of the a3r2 is "1176842.5189999999".

[create a4r2]
a4r2 is a area. "An area (4) in r2".
the printed name of the a4r2 is "Room 2".
Understand "Area 4 in Room 2" as a4r2.
the parent of the a4r2 is r2.
the X of the a4r2 is "547349.0".
the Y of the a4r2 is "1176824.0".

[create a5r2]
a5r2 is a area. "An area (5) in r2".
the printed name of the a5r2 is "Room 2".
Understand "Area 5 in Room 2" as a5r2.
the parent of the a5r2 is r2.
the X of the a5r2 is "547308.5982564338".
the Y of the a5r2 is "1176842.6848256432".

[create a10r2]
a10r2 is a area. "An area (10) in r2".
the printed name of the a10r2 is "Room 2".
Understand "Area 10 in Room 2" as a10r2.
the parent of the a10r2 is r2.
the X of the a10r2 is "546520.7581967213".
the Y of the a10r2 is "1176861.5819672132".

[create a11r2]
a11r2 is a area. "An area (11) in r2".
the printed name of the a11r2 is "Room 2".
Understand "Area 11 in Room 2" as a11r2.
the parent of the a11r2 is r2.
the X of the a11r2 is "546556.8954988092".
the Y of the a11r2 is "1176884.4393435614".

[create a12r2]
a12r2 is a area. "An area (12) in r2".
the printed name of the a12r2 is "Room 2".
Understand "Area 12 in Room 2" as a12r2.
the parent of the a12r2 is r2.
the X of the a12r2 is "545120.0".
the Y of the a12r2 is "1176914.0".

[create a13r2]
a13r2 is a area. "An area (13) in r2".
the printed name of the a13r2 is "Room 2".
Understand "Area 13 in Room 2" as a13r2.
the parent of the a13r2 is r2.
the X of the a13r2 is "545150.9153780071".
the Y of the a13r2 is "1176888.1834621993".

[create a14r2]
a14r2 is a area. "An area (14) in r2".
the printed name of the a14r2 is "Room 2".
Understand "Area 14 in Room 2" as a14r2.
the parent of the a14r2 is r2.
the X of the a14r2 is "547340.0".
the Y of the a14r2 is "1176914.0".

[create a15r2]
a15r2 is a area. "An area (15) in r2".
the printed name of the a15r2 is "Room 2".
Understand "Area 15 in Room 2" as a15r2.
the parent of the a15r2 is r2.
the X of the a15r2 is "547309.0846219931".
the Y of the a15r2 is "1176888.1834621993".

[create a26r2]
a26r2 is a area. "An area (26) in r2".
the printed name of the a26r2 is "Room 2".
Understand "Area 26 in Room 2" as a26r2.
the parent of the a26r2 is r2.
the X of the a26r2 is "546695.8043478262".
the Y of the a26r2 is "1176775.0".

[create a27r2]
a27r2 is a area. "An area (27) in r2".
the printed name of the a27r2 is "Room 2".
Understand "Area 27 in Room 2" as a27r2.
the parent of the a27r2 is r2.
the X of the a27r2 is "546696.1329431438".
the Y of the a27r2 is "1176824.4502508359".

[create a32r2]
a32r2 is a area. "An area (32) in r2".
the printed name of the a32r2 is "Room 2".
Understand "Area 32 in Room 2" as a32r2.
the parent of the a32r2 is r2.
the X of the a32r2 is "546704.4815264789".
the Y of the a32r2 is "1176840.184119429".

[create a33r2]
a33r2 is a area. "An area (33) in r2".
the printed name of the a33r2 is "Room 2".
Understand "Area 33 in Room 2" as a33r2.
the parent of the a33r2 is r2.
the X of the a33r2 is "545168.092671901".
the Y of the a33r2 is "1176859.8065726862".

[create a34r2]
a34r2 is a area. "An area (34) in r2".
the printed name of the a34r2 is "Room 2".
Understand "Area 34 in Room 2" as a34r2.
the parent of the a34r2 is r2.
the X of the a34r2 is "547292.6927655565".
the Y of the a34r2 is "1176861.1041153662".

[create a35r2]
a35r2 is a area. "An area (35) in r2".
the printed name of the a35r2 is "Room 2".
Understand "Area 35 in Room 2" as a35r2.
the parent of the a35r2 is r2.
the X of the a35r2 is "546588.984698933".
the Y of the a35r2 is "1176838.736570449".

[create a41r2]
a41r2 is a area. "An area (41) in r2".
the printed name of the a41r2 is "Room 2".
Understand "Area 41 in Room 2" as a41r2.
the parent of the a41r2 is r2.
the X of the a41r2 is "545527.6980415579".
the Y of the a41r2 is "1176733.2713725315".

[create a42r2]
a42r2 is a area. "An area (42) in r2".
the printed name of the a42r2 is "Room 2".
Understand "Area 42 in Room 2" as a42r2.
the parent of the a42r2 is r2.
the X of the a42r2 is "546923.6942110015".
the Y of the a42r2 is "1176748.1635622135".

[create d0]
d0 is a door. "A door between a39r0 and a37r1".
d0 is south of a39r0 and north of a37r1.

[create d2]
d2 is a door. "A door between a40r1 and a41r2".
d2 is north of a40r1 and south of a41r2.

[create d3]
d3 is a door. "A door between a43r1 and a42r2".
d3 is north of a43r1 and south of a42r2.

southeast of a0r0 is northwest of a1r0.

northeast of a2r2 is southwest of a3r2.

northwest of a4r2 is southeast of a5r2.

northeast of a6r0 is southwest of a7r0.

northwest of a8r0 is southeast of a9r0.

northeast of a10r2 is southwest of a11r2.

southeast of a12r2 is northwest of a13r2.

southwest of a14r2 is northeast of a15r2.

southwest of a16r0 is northeast of a17r0.

northeast of a18r1 is southwest of a19r1.

southeast of a20r1 is northwest of a21r1.

northwest of a22r1 is southeast of a23r1.

northwest of a24r1 is southeast of a25r1.

north of a26r2 is south of a27r2.

northeast of a28r1 is southwest of a29r1.

southwest of a17r0 is northeast of a30r0.

southeast of a1r0 is northwest of a31r0.

northeast of a27r2 is southwest of a32r2.

northeast of a3r2 is southwest of a33r2.

northwest of a5r2 is southeast of a34r2.

northeast of a7r0 is southwest of a31r0.

northwest of a9r0 is southeast of a30r0.

southeast of a11r2 is northwest of a35r2.

southeast of a13r2 is northwest of a33r2.

southwest of a15r2 is northeast of a34r2.

northeast of a19r1 is southwest of a36r1.

southeast of a21r1 is northwest of a36r1.

northwest of a23r1 is southeast of a37r1.

northwest of a25r1 is southeast of a38r1.

west of a30r0 is east of a39r0.

west of a35r2 is east of a41r2.

east of a33r2 is west of a41r2.

southeast of a32r2 is northwest of a42r2.

west of a34r2 is east of a42r2.

east of a31r0 is west of a39r0.

west of a32r2 is east of a35r2.

east of a36r1 is west of a40r1.

west of a38r1 is east of a40r1.

west of a37r1 is east of a44r1.

northeast of a29r1 is southwest of a44r1.

northeast of a38r1 is southwest of a43r1.

southeast of a43r1 is northwest of a44r1.

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
the player is in a16r0.

the orientation of the player is 6.
