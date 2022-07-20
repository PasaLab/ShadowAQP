import sqlparse
from sqlparse.tokens import Token
import copy


class TableAttribute:
    def __init__(self, table_name, attr_name):
        self.attr_name = attr_name
        self.table_name = table_name
    def __str__(self):
        return self.table_name+"."+self.attr_name

class Table:
    """Represents a table with foreign key and primary key relationships"""

    def __init__(self, database_name, table_name, categorical_attributes=[], numeric_attributes=[],
                 table_size=1000, csv_file_location=None, sample_rate=1.0):
        self.database_name = database_name
        self.table_name = table_name
        self.table_size = table_size
        self.categorical_attributes = categorical_attributes
        self.numeric_attributes = numeric_attributes
        self.attributes = categorical_attributes + numeric_attributes
        self.sample_rate = sample_rate


class QueryTable:
    def __init__(self, database_name, table_name):
        self.database_name = database_name
        self.table_name = table_name
        self.qualified_table_name=database_name+"."+table_name
        self.conditions = []
        self.columns = []
        self.involved_cols=[]
        self.join_col = ''
        self.sum_cols = []
        self.avg_cols = []
        self.agg_cols = []
        self.aggregations = []
        self.group_bys = []
        self.group_cols=[]
        self.sampling_method=''
        self.sampling_rate = 0.001
        self.config_file=''
        self.rate_col=''
        self.group_num_col=''
        self.sample_num_col=''
        self.num_col=''
        self.outlier_flag=False
        self.group_info=None
        self.group_var_info=None
        self.bound_info=None
        self.sample_info=None
        # self.count_info=None

    def add_aggregation(self,operation):
        self.aggregations.append(operation)
        aggregation_attr=operation[1]
        if operation[0]=='SUM':
            self.sum_cols.append(aggregation_attr.attr_name)
        elif operation[0]=='AVG':
            self.avg_cols.append(aggregation_attr.attr_name)
        if aggregation_attr.attr_name not in self.agg_cols:
            self.agg_cols.append(aggregation_attr.attr_name)
        if aggregation_attr.attr_name not in self.involved_cols:
            self.involved_cols.append(aggregation_attr.attr_name)

    def add_sampling_method(self,method):
        splits=method.split('-')
        self.sampling_method=splits[0]
        self.sampling_rate=float(splits[1])
        self.config_file=splits[2].lower()

    def __str__(self):
        aggregations = '\n'.join(
            [str(agg[0]) + ":" + str(agg[1].attr_name) for agg in self.aggregations])
        
        # join_condition = '='.join(self.join_condition)
        # conditions = ','.join(self.conditions)
        # group_bys = ','.join(self.group_bys)
        return "Table Info:\ndatabase:{}\ntable:{}\ncolumns:{}\ninvolved_cols:{}\ngroup_cols:{}\naggregations:{}\njoin_column:{}\ncondition:{}\ngroup_bys:{}\nsampling_method:{}\nsampling_rate:{}".format(self.database_name, self.table_name,self.columns,
                                                                                                                 self.involved_cols,self.group_cols,aggregations, self.join_col, self.conditions, self.group_bys,self.sampling_method,self.sampling_rate)


class Schema:
    """Holds all tables and relationships"""

    def __init__(self):
        self.tables = []
        self.table_dictionary = {}

    def add_table(self, table):
        self.tables.append(table)
        self.table_dictionary[table.table_name] = table

    def get_table(self, table_name):
        return self.table_dictionary[table_name]


class Query:
    """Represents query"""

    def __init__(self):
        self.sql=""
        self.databases = []
        self.tables = []
        self.table_where_condition_dict = {}
        self.conditions = []
        self.join_condition = None
        self.aggregation_operations = []
        self.group_bys = []
        self.query_tables_dict = {}
        self.query_tables = []
        self.multi_sampling_times=1

    def add_table(self, query_table):
        table_name = query_table.table_name
        database_name = query_table.database_name
        self.databases.append(database_name)
        self.tables.append(table_name)
        self.query_tables_dict[table_name] = query_table
        self.query_tables.append(query_table)

    def remove_conditions_for_attributes(self, table, attributes):
        def conflicting(condition):
            return any([condition.startswith(attribute + ' ') or condition.startswith(attribute + '<') or
                        condition.startswith(attribute + '>') or condition.startswith(attribute + '=') for
                        attribute in attributes])

        if self.table_where_condition_dict.get(table) is not None:
            self.table_where_condition_dict[table] = [condition for condition in
                                                      self.table_where_condition_dict[table]
                                                      if not conflicting(condition)]
        self.conditions = [(cond_table, condition) for cond_table, condition in self.conditions
                           if not (cond_table == table and conflicting(condition))]

    def add_group_by(self, attribute):
        self.group_bys.append(attribute)
        self.query_tables_dict[attribute.table_name].group_bys.append(
            attribute.attr_name)
        if attribute.attr_name not in self.query_tables_dict[attribute.table_name].involved_cols:
            self.query_tables_dict[attribute.table_name].involved_cols.append(attribute.attr_name)
        if attribute.attr_name not in self.query_tables_dict[attribute.table_name].group_cols:
            self.query_tables_dict[attribute.table_name].group_cols.append(attribute.attr_name)

    def add_aggregation_operation(self, operation):
        """
        Adds operation to AQP query.
        :param operation: (AggregationOperationType.AGGREGATION, operation_type, operation_factors) or (AggregationOperationType.MINUS, None, None)
        :return:
        """
        self.aggregation_operations.append(operation)
        aggregation_attr=operation[1]
        self.query_tables_dict[aggregation_attr.table_name].add_aggregation(operation)

    def add_join_condition(self, left, right):
        self.join_condition = (left, right)
        self.query_tables_dict[left.table_name].join_col = left.attr_name
        if left.attr_name not in self.query_tables_dict[left.table_name].involved_cols:
            self.query_tables_dict[left.table_name].involved_cols.append(left.attr_name)
        if left.attr_name not in self.query_tables_dict[left.table_name].group_cols:
            self.query_tables_dict[left.table_name].group_cols.append(left.attr_name)
        self.query_tables_dict[right.table_name].join_col = right.attr_name
        if right.attr_name not in self.query_tables_dict[right.table_name].involved_cols:
            self.query_tables_dict[right.table_name].involved_cols.append(right.attr_name)
        if right.attr_name not in self.query_tables_dict[right.table_name].group_cols:
            self.query_tables_dict[right.table_name].group_cols.append(right.attr_name)

    def add_where_condition(self, table,attr_name, condition):
        if self.table_where_condition_dict.get(table) is None:
            self.table_where_condition_dict[table] = [condition]
        else:
            self.table_where_condition_dict[table].append(condition)
        self.conditions.append((table, condition))
        self.query_tables_dict[table].conditions.append(condition)
        if attr_name not in self.query_tables_dict[table].involved_cols:
            self.query_tables_dict[table].involved_cols.append(attr_name)
        if attr_name not in self.query_tables_dict[table].group_cols:
            self.query_tables_dict[table].group_cols.append(attr_name)

    def __str__(self):
        tables = [t[0] + "." + t[1] for t in zip(self.databases, self.tables)]
        aggregations = '\n'.join(
            [str(agg[0]) + ":" + str(agg[1].attr_name) for agg in self.aggregation_operations])
        join_condition = str(self.join_condition[0])+"="+str(self.join_condition[1])
        # conditions = ','.join(self.conditions)
        group_bys = [str(t) for t in self.group_bys]
        return "Query Info:\ntables:{}\naggregations:{}\njoin_condition:{}\ncondition:{}\ngroup_bys:{}".format(tables, aggregations,
                                                                                                  join_condition,
                                                                                                  self.conditions,
                                                                                                  group_bys)
