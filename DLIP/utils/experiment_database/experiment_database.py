import sqlite3
import logging
from sqlite3.dbapi2 import Error
from contextlib import closing

from DLIP.utils.helper_functions.encode_64 import encodeb64


class ExperimentDatabase():
    def __init__(self, db_file):
        self.conn = None
        try:
            self.conn = sqlite3.connect(db_file)
            self.create_pretexts_if_non_existing()
            self.create_downstreams_if_non_existing()
        except Error as e:
            logging.error(e)
            
    @property
    def create_pretext_task_command(self):
        return """ CREATE TABLE IF NOT EXISTS Pretexts (
                                            id integer PRIMARY KEY AUTOINCREMENT,
                                            experiment_name text,
                                            name text,
                                            dataset text,
                                            ssl_method text,
                                            create_time REAL DEFAULT (datetime('now', 'localtime')),
                                            experiment_dir text,
                                            checkpoint_path text,
                                            config blob,
                                            finished integer
                                        ); """
                                        
    @property
    def create_downstream_taks_command(self):
        return """ CREATE TABLE IF NOT EXISTS Downstreams (
                                            id integer PRIMARY KEY AUTOINCREMENT,
                                            experiment_name text,
                                            name text,
                                            dataset text,
                                            sl_method text,
                                            create_time REAL DEFAULT (datetime('now', 'localtime')),
                                            experiment_dir text,
                                            checkpoint_path text,
                                            config blob,
                                            finished integer,
                                            pretext_id integer,
                                            FOREIGN KEY (pretext_id) REFERENCES Pretexts (id)
                                        ); """

    def create_pretexts_if_non_existing(self):
        try:
            with closing(self.conn.cursor()) as c:
                c.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='Pretexts' ''')
                #if the count is 1, then table exists
                if c.fetchone()[0]!=1 : 
                    logging.info('Pretexts Table does not exist.')
                    c.execute(self.create_pretext_task_command)
                    self.conn.commit()
        except Error as e:
                logging.error(e)

    def create_downstreams_if_non_existing(self):
        try:
            with closing(self.conn.cursor()) as c:
                c.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='Downstreams' ''')
                #if the count is 1, then table exists
                if c.fetchone()[0]!=1 : 
                    logging.info('Downstreams Table does not exist.')
                    c.execute(self.create_downstream_taks_command)
                    self.conn.commit()
        except Error as e:
            logging.error(f'ERROR: {e}')
    
    def insert_pretext(self, experiment_name, name, dataset, ssl_method, experiment_dir, checkpoint_path, config):
        sql = f' INSERT INTO Pretexts(experiment_name, name, dataset, ssl_method, experiment_dir, checkpoint_path, config,finished) VALUES(?,?,?,?,?,?,?,?)'
        try:
            with closing(self.conn.cursor()) as c:
                c.execute(sql,
                    (
                    experiment_name,
                    name,
                    dataset,
                    ssl_method,
                    experiment_dir,
                    checkpoint_path,
                    encodeb64(config),
                    0
                ))
                self.conn.commit()
                return c.lastrowid
        except Error as e:
            logging.error(f'ERROR: {e}')

    def insert_downstream(self, name, dataset, ssl_method, checkpoint_path, config, pretext_id):
        sql = f' INSERT INTO Downstreams(name, dataset, sl_method, checkpoint_path, config, pretext_id, finished) VALUES(?,?,?,?,?,?,?)'
        try:
            with closing(self.conn.cursor()) as c:
                c.execute(sql,
                    (name,
                    dataset,
                    ssl_method,
                    checkpoint_path,
                    encodeb64(config),
                    pretext_id,
                    0
            ))
                self.conn.commit()
                return c.lastrowid
        except Error as e:
            logging.error(f'ERROR: {e}')


    def get_pretext_id_by_name(self,name):
        sql = f"SELECT id from Pretexts WHERE NAME LIKE ?"
        try:
            with closing(self.conn.cursor()) as c:
                c.execute(sql,(name,))
                res = c.fetchall()
                if len(res) == 0 :
                    logging.info(f'Pretext {name} does not exist.')
                    return None
                return res[0][0]
        except Error as e:
            logging.error(f'ERROR: {e}')
    
    def update_pretext_status(self,id:int,finished:bool):
        sql = f"UPDATE Pretexts SET finished = ? WHERE id = ? "
        try:
            with closing(self.conn.cursor()) as c:
                c.execute(sql,(finished,id))
                self.conn.commit()
        except Error as e:
                logging.error(e) 
