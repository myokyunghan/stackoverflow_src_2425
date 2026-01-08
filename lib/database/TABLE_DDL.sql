-- public.tt_posts_difficulty definition

-- Drop table

-- DROP TABLE public.tt_posts_difficulty;

CREATE TABLE public.tt_posts_difficulty (
	ver int4 NOT NULL,
	creationdate timestamp NOT NULL,
	id int4 NOT NULL,
	difficulty_level int4 NULL,
	CONSTRAINT tt_posts_difficulty_pkey PRIMARY KEY (ver, creationdate, id)
);

-- Permissions

ALTER TABLE public.tt_posts_difficulty OWNER TO cslab;
GRANT ALL ON TABLE public.tt_posts_difficulty TO cslab;


-- public.tt_posts_difficulty_annotated definition

-- Drop table

-- DROP TABLE public.tt_posts_difficulty_annotated;

CREATE TABLE public.tt_posts_difficulty_annotated (
	id int4 NOT NULL,
	CONSTRAINT tt_posts_difficulty_annotated_pkey PRIMARY KEY (id)
);

-- Permissions

ALTER TABLE public.tt_posts_difficulty_annotated OWNER TO cslab;
GRANT ALL ON TABLE public.tt_posts_difficulty_annotated TO cslab;


-- public.tt_posts_difficulty_done definition

-- Drop table

-- DROP TABLE public.tt_posts_difficulty_done;

CREATE TABLE public.tt_posts_difficulty_done (
	ver int4 NOT NULL,
	creationdate timestamp NOT NULL,
	id int4 NOT NULL,
	CONSTRAINT tt_posts_difficulty_done_pkey PRIMARY KEY (ver, creationdate, id)
);
CREATE INDEX idx_tt_posts_difficulty_done_01 ON public.tt_posts_difficulty_done USING btree (ver, creationdate);

-- Permissions

ALTER TABLE public.tt_posts_difficulty_done OWNER TO sopjt;
GRANT ALL ON TABLE public.tt_posts_difficulty_done TO sopjt;


-- public.tt_posts_difficulty_target definition

-- Drop table

-- DROP TABLE public.tt_posts_difficulty_target;

CREATE TABLE public.tt_posts_difficulty_target (
	ver int4 NOT NULL,
	creationdate timestamp NOT NULL,
	id int4 NOT NULL,
	CONSTRAINT tt_posts_difficulty_target_pkey PRIMARY KEY (ver, creationdate, id)
);
CREATE INDEX idx_tt_posts_difficulty_target_01 ON public.tt_posts_difficulty_target USING btree (ver, creationdate);

-- Permissions

ALTER TABLE public.tt_posts_difficulty_target OWNER TO sopjt;
GRANT ALL ON TABLE public.tt_posts_difficulty_target TO sopjt;