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

[create a5r0]
a5r0 is a area. "An area (5) in r0".
the printed name of the a5r0 is "Room 0".
Understand "Area 5 in Room 0" as a5r0.
the parent of the a5r0 is r0.
the X of the a5r0 is "545261.3749999998".
the Y of the a5r0 is "2161201.379310345".

[create a6r0]
a6r0 is a area. "An area (6) in r0".
the printed name of the a6r0 is "Room 0".
Understand "Area 6 in Room 0" as a6r0.
the parent of the a6r0 is r0.
the X of the a6r0 is "545370.3987047074".
the Y of the a6r0 is "2161094.3342435663".

[create a7r0]
a7r0 is a area. "An area (7) in r0".
the printed name of the a7r0 is "Room 0".
Understand "Area 7 in Room 0" as a7r0.
the parent of the a7r0 is r0.
the X of the a7r0 is "545479.6250000001".
the Y of the a7r0 is "2161202.5".

[create a40r0]
a40r0 is a area. "An area (40) in r0".
the printed name of the a40r0 is "Room 0".
Understand "Area 40 in Room 0" as a40r0.
the parent of the a40r0 is r0.
the X of the a40r0 is "545225.0".
the Y of the a40r0 is "2161240.0".

[create a41r0]
a41r0 is a area. "An area (41) in r0".
the printed name of the a41r0 is "Room 0".
Understand "Area 41 in Room 0" as a41r0.
the parent of the a41r0 is r0.
the X of the a41r0 is "545516.0".
the Y of the a41r0 is "2161240.0".

[create a14r1]
a14r1 is a area. "An area (14) in r1".
the printed name of the a14r1 is "Room 1".
Understand "Area 14 in Room 1" as a14r1.
the parent of the a14r1 is r1.
the X of the a14r1 is "545418.8902277271".
the Y of the a14r1 is "2159946.0875734263".

[create a17r1]
a17r1 is a area. "An area (17) in r1".
the printed name of the a17r1 is "Room 1".
Understand "Area 17 in Room 1" as a17r1.
the parent of the a17r1 is r1.
the X of the a17r1 is "545370.4748096995".
the Y of the a17r1 is "2159145.737436124".

[create a18r1]
a18r1 is a area. "An area (18) in r1".
the printed name of the a18r1 is "Room 1".
Understand "Area 18 in Room 1" as a18r1.
the parent of the a18r1 is r1.
the X of the a18r1 is "545479.625".
the Y of the a18r1 is "2159038.3999999994".

[create a24r1]
a24r1 is a area. "An area (24) in r1".
the printed name of the a24r1 is "Room 1".
Understand "Area 24 in Room 1" as a24r1.
the parent of the a24r1 is r1.
the X of the a24r1 is "545261.375".
the Y of the a24r1 is "2159038.620689655".

[create a44r1]
a44r1 is a area. "An area (44) in r1".
the printed name of the a44r1 is "Room 1".
Understand "Area 44 in Room 1" as a44r1.
the parent of the a44r1 is r1.
the X of the a44r1 is "545225.0".
the Y of the a44r1 is "2159000.0".

[create a45r1]
a45r1 is a area. "An area (45) in r1".
the printed name of the a45r1 is "Room 1".
Understand "Area 45 in Room 1" as a45r1.
the parent of the a45r1 is r1.
the X of the a45r1 is "545516.0".
the Y of the a45r1 is "2159000.0".

[create a8r2]
a8r2 is a area. "An area (8) in r2".
the printed name of the a8r2 is "Room 2".
Understand "Area 8 in Room 2" as a8r2.
the parent of the a8r2 is r2.
the X of the a8r2 is "546667.0220437592".
the Y of the a8r2 is "2161100.6674436205".

[create a9r2]
a9r2 is a area. "An area (9) in r2".
the printed name of the a9r2 is "Room 2".
Understand "Area 9 in Room 2" as a9r2.
the parent of the a9r2 is r2.
the X of the a9r2 is "546963.5".
the Y of the a9r2 is "2161225.635300561".

[create a15r2]
a15r2 is a area. "An area (15) in r2".
the printed name of the a15r2 is "Room 2".
Understand "Area 15 in Room 2" as a15r2.
the parent of the a15r2 is r2.
the X of the a15r2 is "545917.9961837009".
the Y of the a15r2 is "2160292.273223205".

[create a37r2]
a37r2 is a area. "An area (37) in r2".
the printed name of the a37r2 is "Room 2".
Understand "Area 37 in Room 2" as a37r2.
the parent of the a37r2 is r2.
the X of the a37r2 is "546608.0".
the Y of the a37r2 is "2161240.0".

[create a38r2]
a38r2 is a area. "An area (38) in r2".
the printed name of the a38r2 is "Room 2".
Understand "Area 38 in Room 2" as a38r2.
the parent of the a38r2 is r2.
the X of the a38r2 is "546647.4999999995".
the Y of the a38r2 is "2161147.034466668".

[create a42r2]
a42r2 is a area. "An area (42) in r2".
the printed name of the a42r2 is "Room 2".
Understand "Area 42 in Room 2" as a42r2.
the parent of the a42r2 is r2.
the X of the a42r2 is "547003.0".
the Y of the a42r2 is "2161240.0".

[create a10r3]
a10r3 is a area. "An area (10) in r3".
the printed name of the a10r3 is "Room 3".
Understand "Area 10 in Room 3" as a10r3.
the parent of the a10r3 is r3.
the X of the a10r3 is "546595.551792616".
the Y of the a10r3 is "2159461.565370073".

[create a11r3]
a11r3 is a area. "An area (11) in r3".
the printed name of the a11r3 is "Room 3".
Understand "Area 11 in Room 3" as a11r3.
the parent of the a11r3 is r3.
the X of the a11r3 is "546770.9146651698".
the Y of the a11r3 is "2159670.57921053".

[create a12r3]
a12r3 is a area. "An area (12) in r3".
the printed name of the a12r3 is "Room 3".
Understand "Area 12 in Room 3" as a12r3.
the parent of the a12r3 is r3.
the X of the a12r3 is "547481.2385255077".
the Y of the a12r3 is "2160024.0".

[create a13r3]
a13r3 is a area. "An area (13) in r3".
the printed name of the a13r3 is "Room 3".
Understand "Area 13 in Room 3" as a13r3.
the parent of the a13r3 is r3.
the X of the a13r3 is "547465.3631724012".
the Y of the a13r3 is "2159521.8638442187".

[create a16r3]
a16r3 is a area. "An area (16) in r3".
the printed name of the a16r3 is "Room 3".
Understand "Area 16 in Room 3" as a16r3.
the parent of the a16r3 is r3.
the X of the a16r3 is "548442.7448661658".
the Y of the a16r3 is "2160048.2093226733".

[create a19r3]
a19r3 is a area. "An area (19) in r3".
the printed name of the a19r3 is "Room 3".
Understand "Area 19 in Room 3" as a19r3.
the parent of the a19r3 is r3.
the X of the a19r3 is "546761.6414363468".
the Y of the a19r3 is "2159199.3710393733".

[create a20r3]
a20r3 is a area. "An area (20) in r3".
the printed name of the a20r3 is "Room 3".
Understand "Area 20 in Room 3" as a20r3.
the parent of the a20r3 is r3.
the X of the a20r3 is "547332.2322606249".
the Y of the a20r3 is "2159167.822884385".

[create a21r3]
a21r3 is a area. "An area (21) in r3".
the printed name of the a21r3 is "Room 3".
Understand "Area 21 in Room 3" as a21r3.
the parent of the a21r3 is r3.
the X of the a21r3 is "546688.5".
the Y of the a21r3 is "2159066.893870239".

[create a22r3]
a22r3 is a area. "An area (22) in r3".
the printed name of the a22r3 is "Room 3".
Understand "Area 22 in Room 3" as a22r3.
the parent of the a22r3 is r3.
the X of the a22r3 is "547426.0191403141".
the Y of the a22r3 is "2159328.270284978".

[create a23r3]
a23r3 is a area. "An area (23) in r3".
the printed name of the a23r3 is "Room 3".
Understand "Area 23 in Room 3" as a23r3.
the parent of the a23r3 is r3.
the X of the a23r3 is "547453.5".
the Y of the a23r3 is "2159021.869377163".

[create a25r3]
a25r3 is a area. "An area (25) in r3".
the printed name of the a25r3 is "Room 3".
Understand "Area 25 in Room 3" as a25r3.
the parent of the a25r3 is r3.
the X of the a25r3 is "547610.0".
the Y of the a25r3 is "2159159.0".

[create a26r3]
a26r3 is a area. "An area (26) in r3".
the printed name of the a26r3 is "Room 3".
Understand "Area 26 in Room 3" as a26r3.
the parent of the a26r3 is r3.
the X of the a26r3 is "547527.7383533369".
the Y of the a26r3 is "2159231.5666468847".

[create a27r3]
a27r3 is a area. "An area (27) in r3".
the printed name of the a27r3 is "Room 3".
Understand "Area 27 in Room 3" as a27r3.
the parent of the a27r3 is r3.
the X of the a27r3 is "547744.0".
the Y of the a27r3 is "2159463.0".

[create a28r3]
a28r3 is a area. "An area (28) in r3".
the printed name of the a28r3 is "Room 3".
Understand "Area 28 in Room 3" as a28r3.
the parent of the a28r3 is r3.
the X of the a28r3 is "547583.3641791053".
the Y of the a28r3 is "2159497.5".

[create a29r3]
a29r3 is a area. "An area (29) in r3".
the printed name of the a29r3 is "Room 3".
Understand "Area 29 in Room 3" as a29r3.
the parent of the a29r3 is r3.
the X of the a29r3 is "548511.0".
the Y of the a29r3 is "2159896.0".

[create a30r3]
a30r3 is a area. "An area (30) in r3".
the printed name of the a30r3 is "Room 3".
Understand "Area 30 in Room 3" as a30r3.
the parent of the a30r3 is r3.
the X of the a30r3 is "548472.0833333334".
the Y of the a30r3 is "2159982.063303571".

[create a43r3]
a43r3 is a area. "An area (43) in r3".
the printed name of the a43r3 is "Room 3".
Understand "Area 43 in Room 3" as a43r3.
the parent of the a43r3 is r3.
the X of the a43r3 is "547480.0".
the Y of the a43r3 is "2159000.0".

[create a46r3]
a46r3 is a area. "An area (46) in r3".
the printed name of the a46r3 is "Room 3".
Understand "Area 46 in Room 3" as a46r3.
the parent of the a46r3 is r3.
the X of the a46r3 is "546659.0".
the Y of the a46r3 is "2159000.0".

[create a47r3]
a47r3 is a area. "An area (47) in r3".
the printed name of the a47r3 is "Room 3".
Understand "Area 47 in Room 3" as a47r3.
the parent of the a47r3 is r3.
the X of the a47r3 is "546797.0".
the Y of the a47r3 is "2159704.0".

[create a0r4]
a0r4 is a area. "An area (0) in r4".
the printed name of the a0r4 is "Room 4".
Understand "Area 0 in Room 4" as a0r4.
the parent of the a0r4 is r4.
the X of the a0r4 is "548717.6480764118".
the Y of the a0r4 is "2160352.7637692993".

[create a1r4]
a1r4 is a area. "An area (1) in r4".
the printed name of the a1r4 is "Room 4".
Understand "Area 1 in Room 4" as a1r4.
the parent of the a1r4 is r4.
the X of the a1r4 is "548614.2303650749".
the Y of the a1r4 is "2160931.286401533".

[create a2r4]
a2r4 is a area. "An area (2) in r4".
the printed name of the a2r4 is "Room 4".
Understand "Area 2 in Room 4" as a2r4.
the parent of the a2r4 is r4.
the X of the a2r4 is "548465.2618692743".
the Y of the a2r4 is "2161062.4624382".

[create a3r4]
a3r4 is a area. "An area (3) in r4".
the printed name of the a3r4 is "Room 4".
Understand "Area 3 in Room 4" as a3r4.
the parent of the a3r4 is r4.
the X of the a3r4 is "547466.2930316457".
the Y of the a3r4 is "2160961.717860191".

[create a4r4]
a4r4 is a area. "An area (4) in r4".
the printed name of the a4r4 is "Room 4".
Understand "Area 4 in Room 4" as a4r4.
the parent of the a4r4 is r4.
the X of the a4r4 is "547221.9999999999".
the Y of the a4r4 is "2161201.8461538465".

[create a31r4]
a31r4 is a area. "An area (31) in r4".
the printed name of the a31r4 is "Room 4".
Understand "Area 31 in Room 4" as a31r4.
the parent of the a31r4 is r4.
the X of the a31r4 is "548868.0".
the Y of the a31r4 is "2160296.0".

[create a32r4]
a32r4 is a area. "An area (32) in r4".
the printed name of the a32r4 is "Room 4".
Understand "Area 32 in Room 4" as a32r4.
the parent of the a32r4 is r4.
the X of the a32r4 is "548770.0110044023".
the Y of the a32r4 is "2160332.125".

[create a33r4]
a33r4 is a area. "An area (33) in r4".
the printed name of the a33r4 is "Room 4".
Understand "Area 33 in Room 4" as a33r4.
the parent of the a33r4 is r4.
the X of the a33r4 is "548734.0".
the Y of the a33r4 is "2161035.0".

[create a34r4]
a34r4 is a area. "An area (34) in r4".
the printed name of the a34r4 is "Room 4".
Understand "Area 34 in Room 4" as a34r4.
the parent of the a34r4 is r4.
the X of the a34r4 is "548625.0919555".
the Y of the a34r4 is "2160940.040149067".

[create a35r4]
a35r4 is a area. "An area (35) in r4".
the printed name of the a35r4 is "Room 4".
Understand "Area 35 in Room 4" as a35r4.
the parent of the a35r4 is r4.
the X of the a35r4 is "548515.0".
the Y of the a35r4 is "2161191.0".

[create a36r4]
a36r4 is a area. "An area (36) in r4".
the printed name of the a36r4 is "Room 4".
Understand "Area 36 in Room 4" as a36r4.
the parent of the a36r4 is r4.
the X of the a36r4 is "548466.1699790794".
the Y of the a36r4 is "2161064.5198744773".

[create a39r4]
a39r4 is a area. "An area (39) in r4".
the printed name of the a39r4 is "Room 4".
Understand "Area 39 in Room 4" as a39r4.
the parent of the a39r4 is r4.
the X of the a39r4 is "547189.0".
the Y of the a39r4 is "2161240.0".

[create d0]
d0 is a door. "A door between a15r2 and a10r3".
d0 is southeast of a15r2 and northwest of a10r3.

[create d2]
d2 is a door. "A door between a14r1 and a15r2".
d2 is northeast of a14r1 and southwest of a15r2.

[create d4]
d4 is a door. "A door between a6r0 and a14r1".
d4 is south of a6r0 and north of a14r1.

[create d6]
d6 is a door. "A door between a16r3 and a0r4".
d6 is northeast of a16r3 and southwest of a0r4.

[create d7]
d7 is a door. "A door between a12r3 and a3r4".
d7 is north of a12r3 and south of a3r4.

north of a0r4 is south of a1r4.

northwest of a1r4 is southeast of a2r4.

west of a2r4 is east of a3r4.

northwest of a3r4 is southeast of a4r4.

southeast of a5r0 is northwest of a6r0.

northeast of a6r0 is southwest of a7r0.

northeast of a8r2 is southwest of a9r2.

northeast of a10r3 is southwest of a11r3.

south of a12r3 is north of a13r3.

east of a12r3 is west of a16r3.

southwest of a8r2 is northeast of a15r2.

southeast of a17r1 is northwest of a18r1.

east of a19r3 is west of a20r3.

southwest of a19r3 is northeast of a21r3.

south of a14r1 is north of a17r1.

southeast of a10r3 is northwest of a19r3.

northeast of a20r3 is southwest of a22r3.

south of a13r3 is north of a22r3.

southeast of a20r3 is northwest of a23r3.

southwest of a17r1 is northeast of a24r1.

northwest of a25r3 is southeast of a26r3.

west of a27r3 is east of a28r3.

northwest of a29r3 is southeast of a30r3.

west of a31r4 is east of a32r4.

southwest of a33r4 is northeast of a34r4.

south of a35r4 is north of a36r4.

southeast of a37r2 is northwest of a38r2.

northwest of a4r4 is southeast of a39r4.

northwest of a5r0 is southeast of a40r0.

northeast of a7r0 is southwest of a41r0.

east of a9r2 is west of a42r2.

southeast of a23r3 is northwest of a43r3.

southwest of a24r1 is northeast of a44r1.

southeast of a18r1 is northwest of a45r1.

southwest of a21r3 is northeast of a46r3.

northeast of a11r3 is southwest of a47r3.

southeast of a22r3 is northwest of a26r3.

east of a13r3 is west of a28r3.

southeast of a16r3 is northwest of a30r3.

east of a0r4 is west of a32r4.

northeast of a1r4 is southwest of a34r4.

northeast of a2r4 is southwest of a36r4.

northwest of a8r2 is southeast of a38r2.

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
the player is in a27r3.

the orientation of the player is 6.
