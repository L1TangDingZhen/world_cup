CREATE TABLE models (
	id SERIAL NOT NULL, 
	model_version VARCHAR(80) NOT NULL, 
	model_type VARCHAR(80) NOT NULL, 
	home_advantage_gamma FLOAT, 
	rho FLOAT, 
	time_decay_xi FLOAT, 
	competition_weights_json JSON, 
	elo_k_factor FLOAT, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (model_version)
);

CREATE TABLE teams (
	id SERIAL NOT NULL, 
	name VARCHAR(120) NOT NULL, 
	code VARCHAR(12), 
	confederation VARCHAR(32), 
	provider VARCHAR(32), 
	provider_id VARCHAR(64), 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (name)
);

CREATE TABLE matches (
	id SERIAL NOT NULL, 
	date DATE NOT NULL, 
	home_team_id INTEGER NOT NULL, 
	away_team_id INTEGER NOT NULL, 
	home_goals INTEGER, 
	away_goals INTEGER, 
	competition_type VARCHAR(120) NOT NULL, 
	country VARCHAR(120), 
	provider VARCHAR(32), 
	provider_id VARCHAR(64), 
	neutral_venue BOOLEAN NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(home_team_id) REFERENCES teams (id), 
	FOREIGN KEY(away_team_id) REFERENCES teams (id)
);

CREATE TABLE players (
	id SERIAL NOT NULL, 
	name VARCHAR(120) NOT NULL, 
	national_team_id INTEGER, 
	position VARCHAR(32), 
	provider VARCHAR(32), 
	provider_id VARCHAR(64), 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(national_team_id) REFERENCES teams (id)
);

CREATE TABLE rating_snapshots (
	id SERIAL NOT NULL, 
	computed_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	team_id INTEGER NOT NULL, 
	model_version VARCHAR(80) NOT NULL, 
	elo FLOAT, 
	attack FLOAT, 
	defense FLOAT, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(team_id) REFERENCES teams (id)
);

CREATE TABLE simulation_results (
	id SERIAL NOT NULL, 
	simulation_run_id VARCHAR(80) NOT NULL, 
	team_id INTEGER NOT NULL, 
	group_qualify_prob FLOAT NOT NULL, 
	round_of_32_prob FLOAT NOT NULL, 
	round_of_16_prob FLOAT NOT NULL, 
	quarter_final_prob FLOAT NOT NULL, 
	semi_final_prob FLOAT NOT NULL, 
	final_prob FLOAT NOT NULL, 
	champion_prob FLOAT NOT NULL, 
	model_version VARCHAR(80) NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(team_id) REFERENCES teams (id)
);

CREATE TABLE injuries (
	id SERIAL NOT NULL, 
	player_id INTEGER NOT NULL, 
	reported_at DATE NOT NULL, 
	status VARCHAR(80) NOT NULL, 
	expected_return DATE, 
	PRIMARY KEY (id), 
	FOREIGN KEY(player_id) REFERENCES players (id)
);

CREATE TABLE lineups (
	id SERIAL NOT NULL, 
	match_id INTEGER NOT NULL, 
	player_id INTEGER NOT NULL, 
	team_id INTEGER NOT NULL, 
	starter BOOLEAN NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(match_id) REFERENCES matches (id), 
	FOREIGN KEY(player_id) REFERENCES players (id), 
	FOREIGN KEY(team_id) REFERENCES teams (id)
);

CREATE TABLE player_match_stats (
	id SERIAL NOT NULL, 
	player_id INTEGER NOT NULL, 
	match_id INTEGER NOT NULL, 
	minutes_played INTEGER, 
	goals INTEGER, 
	assists INTEGER, 
	xg FLOAT, 
	xa FLOAT, 
	rating FLOAT, 
	PRIMARY KEY (id), 
	FOREIGN KEY(player_id) REFERENCES players (id), 
	FOREIGN KEY(match_id) REFERENCES matches (id)
);

CREATE TABLE predictions (
	id SERIAL NOT NULL, 
	match_id INTEGER, 
	home_team_id INTEGER NOT NULL, 
	away_team_id INTEGER NOT NULL, 
	home_win_prob FLOAT NOT NULL, 
	draw_prob FLOAT NOT NULL, 
	away_win_prob FLOAT NOT NULL, 
	most_likely_score VARCHAR(16) NOT NULL, 
	score_matrix_json JSON, 
	model_version VARCHAR(80) NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(match_id) REFERENCES matches (id), 
	FOREIGN KEY(home_team_id) REFERENCES teams (id), 
	FOREIGN KEY(away_team_id) REFERENCES teams (id)
);

CREATE TABLE squads (
	id SERIAL NOT NULL, 
	team_id INTEGER NOT NULL, 
	player_id INTEGER NOT NULL, 
	as_of_date DATE NOT NULL, 
	available BOOLEAN NOT NULL, 
	attacking_rating FLOAT, 
	defensive_rating FLOAT, 
	PRIMARY KEY (id), 
	FOREIGN KEY(team_id) REFERENCES teams (id), 
	FOREIGN KEY(player_id) REFERENCES players (id)
);
